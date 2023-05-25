# Copyright (c) 2023, NVIDIA CORPORATION.
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

import logging
import queue
import time
import typing

import mrc
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils import rest_server
from morpheus.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)


@register_stage("from-rest")
class RestSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that listens for incoming REST requests on a specified endpoint.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    lines : bool, default False
        If False, the REST server will expect each request to be a JSON array of objects. If True, the REST server will
        expect each request to be a JSON object per line.
    """

    # TODO add configs, and pass-thru for both gunicorn and flask configs
    def __init__(self, c: Config, sleep_time: float = 0.1, lines: bool = False):
        super().__init__(c)
        self._server_proc = None
        self._queue = None
        self._sleep_time = sleep_time
        self._is_running = AtomicInteger(0)
        self._lines = lines

    @property
    def name(self) -> str:
        return "from-rest"

    def supports_cpp_node(self) -> bool:
        return True

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        self._is_running.value = 1
        (self._server_proc, self._queue) = rest_server.start_rest_server()
        self._server_proc.start()

        processing = True
        while (processing):
            # Read as many messages as we can from the queue if it's empty check to see if we should be shutting down
            # It is important that any messages we received that are in the queue are processed before we shutdown since
            # we already returned an OK response to the client.
            df = None
            data = None
            try:
                data = self._queue.get_nowait()
            except queue.Empty:
                if (self._is_running.value == 0 or not self._server_proc.is_alive()):
                    processing = False
                else:
                    time.sleep(self._sleep_time)
            except ValueError as e:
                logger.error(f"Queue closed unexpectedly: {e}")
                processing = False

            if data is not None:
                try:
                    df = cudf.read_json(data, lines=self._lines)
                except Exception as e:
                    print(data)
                    logger.error(f"Failed to convert request data to DataFrame: {e}")

            if df is not None:
                yield MessageMeta(df)

        self._queue.close()

    def _stop(self):
        if self._server_proc is not None:
            logger.debug("Stopping REST server ...")
            self._server_proc.terminate()
            self._server_proc.join()
            logger.debug("REST server stopped")

        self._is_running.value = 0

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        if self._build_cpp_node():
            import morpheus._lib.stages as _stages
            node = _stages.RestSourceStage(builder, self.unique_name, sleep_time=self._sleep_time, lines=self._lines)
        else:
            node = builder.make_source(self.unique_name, self._generate_frames())

        return node, MessageMeta

    def _post_build_single(self, builder: mrc.Builder, out_pair: StreamPair) -> StreamPair:
        src_node = out_pair[0]
        # TODO: This doesn't work its called when the source exits, not when the pipeline is shutting down
        post_node = builder.make_node(self.unique_name + "-post", ops.on_completed(self._stop))
        builder.make_edge(src_node, post_node)

        return super()._post_build_single(builder, (post_node, out_pair[1]))
