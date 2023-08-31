import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair


class InMemSourceXStage(SingleOutputSource):
    """
    InMemorySourceStage subclass that emits whatever you give it and doesn't assume the source data
    is a dataframe.
    """

    def __init__(self, c: Config, data: typing.List[typing.Any]):
        super().__init__(c)
        self._data = data

    @property
    def name(self) -> str:
        return "from-data"

    def supports_cpp_node(self) -> bool:
        return False

    def output_type(self) -> type:
        return type(self._data[0])

    def _emit_data(self) -> typing.Iterator[typing.Any]:
        for x in self._data:
            yield x

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._emit_data())
        return node, type(self._data[0])