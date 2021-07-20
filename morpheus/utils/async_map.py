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
from functools import partial
from time import time

from streamz import Stream
from streamz.core import convert_interval
from streamz.dask import DaskStream
from tornado import gen
from tornado.queues import Queue

logger = logging.getLogger(__name__)


@Stream.register_api()
class async_map(Stream):
    """
    Apply a function to every element in the stream

    Parameters
    ----------
    func : typing.Callable
        The function to call for each mapped value
    *args :
        The arguments to pass to the function.
    **kwargs: dict
        Keyword arguments to pass to func

    Examples
    --------
    >>> source = Stream()
    >>> source.map(lambda x: 2*x).sink(print)
    >>> for i in range(5):
    ...     source.emit(i)
    0
    2
    4
    6
    8
    """
    def __init__(self, upstream, func, stream_name=None, executor=None, *args, **kwargs):
        # this is one of a few stream specific kwargs
        self.executor = executor
        self.kwargs = kwargs
        self.args = args
        self.func = partial(func, *args, **kwargs)

        Stream.__init__(self, upstream, stream_name=stream_name, ensure_io_loop=True)

        if (self.executor is not None):

            loop = self.loop
            executor = self.executor
            f = self.func

            async def get_value_with_executor(x):
                return await loop.run_in_executor(executor, f, x)

            self.func = get_value_with_executor

    async def update(self, x, who=None, metadata=None):
        try:
            self._retain_refs(metadata)
            r = await self.func(x)
            result = r
        except Exception as e:
            logger.exception(e)
            raise
        else:
            emit = await self._emit(result, metadata=metadata)

            return emit
        finally:
            self._release_refs(metadata)


@Stream.register_api()
class time_delay(Stream):
    """Add a time delay to results"""
    _graphviz_shape = 'octagon'

    def __init__(self, upstream, interval, **kwargs):
        self.interval = convert_interval(interval)
        self.queue = Queue()

        kwargs["ensure_io_loop"] = True
        Stream.__init__(self, upstream, **kwargs)

        self.loop.add_callback(self.cb)

    @gen.coroutine
    def cb(self):
        while True:
            q_time, x, metadata = yield self.queue.get()

            duration = self.interval - (time() - q_time)

            if duration > 0:
                yield gen.sleep(duration)

            yield self._emit(x, metadata=metadata)

            self._release_refs(metadata)

    def update(self, x, who=None, metadata=None):
        self._retain_refs(metadata)
        return self.queue.put((time(), x, metadata))


@Stream.register_api()
class scatter_batch(DaskStream):
    """
    Convert local stream to Dask Stream

    All elements flowing through the input will be scattered out to the cluster
    """
    async def update(self, x, who=None, metadata=None):
        from distributed.client import default_client

        client = default_client()

        self._retain_refs(metadata)
        # We need to make sure that x is treated as it is by dask
        # However, client.scatter works internally different for
        # lists and dicts. So we always use a list here to be sure
        # we know the format exactly. We do not use a key to avoid
        # issues like https://github.com/python-streamz/streams/issues/397.
        future_as_list = await client.scatter(x, asynchronous=True, hash=False)

        # emit_awaitables = []

        # for y in future_as_list:
        #     emit_awaitables.extend(self._emit(y))

        f = await self._emit(future_as_list, metadata=metadata)

        self._release_refs(metadata)

        return f


@Stream.register_api()
class switch(Stream):
    def __init__(self, upstream, predicate: typing.Callable[[typing.Any], bool], *args, **kwargs):
        self.predicate = predicate
        stream_name = kwargs.pop("stream_name", None)

        Stream.__init__(self, upstream, stream_name=stream_name)

    async def update(self, x, who=None, metadata=None):
        downstream_idx = self.predicate(x)

        return await self._emit_idx(x, downstream_idx, metadata=metadata)

    async def _emit_idx(self, x, idx: int, metadata=None):

        self.current_value = x
        self.current_metadata = metadata
        if metadata:
            self._retain_refs(metadata, 1)
        else:
            metadata = []

        result = []

        downstream = list(self.downstreams)[idx]

        r = await downstream.update(x, who=self, metadata=metadata)

        if type(r) is list:
            result.extend(r)
        else:
            result.append(r)

        self._release_refs(metadata)

        return [element for element in result if element is not None]

    async def _emit(self, x, metadata=None):
        raise NotImplementedError("Do not use switch _emit directly! Must use _emit_idx()")
