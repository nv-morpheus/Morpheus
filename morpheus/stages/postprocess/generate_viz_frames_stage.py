# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import sys
import typing

import mrc
import mrc.core.operators as ops
import numpy as np
import pandas as pd
import pyarrow as pa
import websockets.legacy.server
from websockets.server import serve

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)


@register_stage("gen-viz", modes=[PipelineModes.NLP], command_args={"deprecated": True})
class GenerateVizFramesStage(SinglePortStage):
    """
    Write out visualization DataFrames.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    out_dir : str
        Output directory to write visualization frames
    overwrite : bool
        Overwrite file if exists

    """

    def __init__(self, c: Config, server_url: str = "0.0.0.0", server_port: int = 8765):
        super().__init__(c)

        self._server_url = server_url
        self._server_port = server_port

        self._first_timestamp = -1
        self._buffers = []
        self._buffer_queue: AsyncIOProducerConsumerQueue = None

        self._replay_buffer = []

    @property
    def name(self) -> str:
        return "gen_viz"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[morpheus.pipeline.messages.MultiResponseMessage, ]
            Accepted input types

        """
        return (MultiResponseMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def round_to_sec(x):
        """
        Round to even seconds second

        Parameters
        ----------
        x : int/float
            Rounding up the value

        Returns
        -------
        int
            Value rounded up

        """
        return int(round(x / 1000.0) * 1000)

    def _to_vis_df(self, x: MultiResponseMessage):

        idx2label = {
            0: 'address',
            1: 'bank_acct',
            2: 'credit_card',
            3: 'email',
            4: 'govt_id',
            5: 'name',
            6: 'password',
            7: 'phone_num',
            8: 'secret_keys',
            9: 'user'
        }

        df = x.get_meta(["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "data"])

        def indent_data(y: str):
            try:
                return json.dumps(json.loads(y), indent=3)
            except:  # noqa: E722
                return y

        df["data"] = df["data"].apply(indent_data)

        probs = x.get_probs_tensor()
        pass_thresh = (probs >= 0.5).any(axis=1)
        max_arg = probs.argmax(axis=1)

        condlist = [pass_thresh]

        choicelist = [max_arg]

        index_sens_info = np.select(condlist, choicelist, default=len(idx2label))

        df["si"] = pd.Series(np.choose(index_sens_info.get(), list(idx2label.values()) + ["none"]).tolist())

        df["ts_round_sec"] = (df["timestamp"] / 1000.0).astype(int) * 1000

        # Return a list of tuples of (ts_round_sec, dataframe)
        return [(key, group) for key, group in df.groupby(df.ts_round_sec)]

    def _write_viz_file(self, x: typing.List[typing.Tuple[int, pd.DataFrame]]):

        curr_timestamp = x[0][0]

        in_df = pd.concat([df for _, df in x], ignore_index=True).sort_values(by=["timestamp"])

        if (self._first_timestamp == -1):
            self._first_timestamp = curr_timestamp

        offset = (curr_timestamp - self._first_timestamp) / 1000

        fn = os.path.join(self._out_dir, "{}.csv".format(offset))

        assert not os.path.exists(fn)

        in_df.to_csv(fn, columns=["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "si", "data"])

    async def start_async(self):
        """
        Launch the Websocket server and asynchronously send messages via Websocket.
        """

        loop = asyncio.get_event_loop()
        self._loop = loop

        self._buffer_queue = AsyncIOProducerConsumerQueue(maxsize=2, loop=loop)

        async def client_connected(websocket: websockets.legacy.server.WebSocketServerProtocol):
            """
            Establishes a connection with the WebSocket server.
            """

            logger.info("Got connection from: {}:{}".format(*websocket.remote_address))

            while True:
                try:
                    next_buffer = await self._buffer_queue.get()
                    await websocket.send(next_buffer.to_pybytes())
                except Closed:
                    break
                except Exception as ex:
                    logger.exception("Error occurred trying to send message over socket", exc_info=ex)

            logger.info("Disconnected from: {}:{}".format(*websocket.remote_address))

        async def run_server():
            """
            Runs Websocket server.
            """

            try:

                async with serve(client_connected, self._server_url, self._server_port) as server:

                    listening_on = [":".join([str(y) for y in x.getsockname()]) for x in server.sockets]
                    listening_on_str = [f"'{x}'" for x in listening_on]

                    logger.info("Websocket server listening at: {}".format(", ".join(listening_on_str)))

                    await self._server_close_event.wait()

                    logger.info("Server shut down")

                logger.info("Server shut down. Is queue empty: {}".format(self._buffer_queue.empty()))
            except Exception as e:
                logger.error("Error during serve", exc_info=e)
                raise

        self._server_task = loop.create_task(run_server())

        self._server_close_event = asyncio.Event(loop=loop)

        await asyncio.sleep(1.0)

        return await super().start_async()

    async def _stop_server(self):

        logger.info("Shutting down queue")

        await self._buffer_queue.close()

        self._server_close_event.set()

        # Wait for it to
        await self._server_task

    def _build_single(self, seg: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        def node_fn(input, output):

            def write_batch(x: MultiResponseMessage):

                sink = pa.BufferOutputStream()

                # This is the timestamp of the earliest message
                t0 = x.get_meta("timestamp").min()

                df = x.get_meta(["timestamp", "src_ip", "dest_ip", "secret_keys", "data"])

                out_df = cudf.DataFrame()

                out_df["dt"] = (df["timestamp"] - t0).astype(np.int32)
                out_df["src"] = df["src_ip"].str.ip_to_int().astype(np.int32)
                out_df["dst"] = df["dest_ip"].str.ip_to_int().astype(np.int32)
                out_df["lvl"] = df["secret_keys"].astype(np.int32)
                out_df["data"] = df["data"]

                array_table = out_df.to_arrow()

                with pa.ipc.new_stream(sink, array_table.schema) as writer:
                    writer.write(array_table)

                out_buf = sink.getvalue()

                # Enqueue the buffer and block until that completes
                asyncio.run_coroutine_threadsafe(self._buffer_queue.put(out_buf), loop=self._loop).result()

            input.pipe(ops.map(write_batch)).subscribe(output)

            logger.info("Gen-viz stage completed. Waiting for shutdown")

            shutdown_future = asyncio.run_coroutine_threadsafe(self._stop_server(), loop=self._loop)

            # Wait for shutdown. Unless we have a debugger attached
            shutdown_future.result(timeout=2.0 if sys.gettrace() is None else None)

            logger.info("Gen-viz shutdown complete")

        # Sink to file
        to_file = seg.make_node(self.unique_name, ops.build(node_fn))
        seg.make_edge(stream, to_file)
        stream = to_file

        # Return input unchanged to allow passthrough
        return input_stream
