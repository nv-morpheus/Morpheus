# Copyright (c) 2021, NVIDIA CORPORATION.
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
import typing
from functools import reduce

import numpy as np
import pandas as pd
import typing_utils
from streamz import Source
from streamz.core import RefCounter
from streamz.core import Stream
from tornado import gen
from tornado.ioloop import IOLoop

import cudf

from morpheus.config import Config
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


@Stream.register_api(staticmethod)
class from_iterable_done(Source):
    """
    Emits items from an iterable.

    Parameters
    ----------
    iterable : iterable
        An iterable to emit messages from.

    Examples
    --------
    >>> source = Stream.from_iterable(range(3))
    >>> L = source.sink_to_list()
    >>> source.start()
    >>> L
    [0, 1, 2]

    """
    def __init__(self, iterable, **kwargs):
        self._iterable = iterable
        super().__init__(**kwargs)

        self._total_count = 0
        self._counters: typing.List[RefCounter] = []

    async def _source_generator(self):
        async for x in self._iterable:
            yield self._emit(x)

            if (self.stopped):
                break

    @gen.coroutine
    def _ref_callback(self):
        self._total_count -= 1

        sum_count = reduce(lambda count, x: count + min(x.count, 1), self._counters, 0)

        if (sum_count != self._total_count):
            logger.debug("Mismatch. Sum: {}, Count: {}".format(sum_count, self._total_count))

    async def _run(self):
        count = 0
        async for x in self._iterable:
            if self.stopped:
                break

            self._total_count += 1

            count += 1

            await self._emit(x)
        self.stopped = True


def df_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. When loading a JSON file, cuDF converts ``\\n`` to
    ``\\\\n`` for some reason
    """

    if ("data" in x):
        x["data"] = x["data"].str.replace('\\n', '\n', regex=False)

    return x


class FileSourceStage(SingleOutputSource):
    """
    This class Load messages from a file and dumps the contents into the pipeline immediately. Useful for
    testing throughput.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be read. Must be JSON lines.

    """
    def __init__(self, c: Config, filename: str, iterative: bool = None):
        super().__init__(c)

        self._filename = filename
        self._batch_size = c.pipeline_batch_size
        self._input_count = None
        self._use_dask = c.use_dask
        self._max_concurrent = c.num_threads

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
        self._iterative = iterative if iterative is not None else not c.use_dask
        self._repeat_count = 5

    @property
    def name(self) -> str:
        return "from-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def _build_source(self) -> typing.Tuple[Source, typing.Type]:

        df = cudf.read_json(self._filename, engine="cudf", lines=True)

        df = df_onread_cleanup(df)

        out_stream: Source = Stream.from_iterable_done(self._generate_frames(df),
                                                       max_concurrent=self._max_concurrent,
                                                       asynchronous=True,
                                                       loop=IOLoop.current())
        out_type = cudf.DataFrame if self._iterative else typing.List[cudf.DataFrame]

        return out_stream, out_type

    def _post_build_single(self, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]
        out_type = out_pair[1]

        # Convert our list of dataframes into the desired type. Either scatter than flatten or just flatten if not using
        # dask
        if (self._use_dask):
            if (typing_utils.issubtype(out_type, typing.List)):
                out_stream = out_stream.scatter_batch().flatten()
                out_type = StreamFuture[typing.get_args(out_type)[0]]
            else:
                out_stream = out_stream.scatter()
                out_type = StreamFuture[out_type]
        else:
            if (typing_utils.issubtype(out_type, typing.List)):
                out_stream = out_stream.flatten()
                out_type = typing.get_args(out_type)[0]

        return super()._post_build_single((out_stream, out_type))

    async def _generate_frames(self, df):
        count = 0
        out = []

        for _ in range(self._repeat_count):
            for x in df.groupby(np.arange(len(df)) // self._batch_size):
                y = x[1].reset_index(drop=True)

                count += 1

                if (self._iterative):
                    yield y
                else:
                    out.append(y)

            if (not self._iterative):
                yield out

        # Indicate that we are stopping (not the best way of doing this)
        self._source_stream.stop()
