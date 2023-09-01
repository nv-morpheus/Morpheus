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

import typing

import mrc

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair


class InMemorySourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Input source that emits a pre-defined list of dataframes.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    dataframes : typing.List[cudf.DataFrame]
        List of dataframes to emit wrapped in `MessageMeta` instances in order
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    """

    def __init__(self, c: Config, dataframes: typing.List[cudf.DataFrame], repeat: int = 1):
        super().__init__(c)

        self._dataframes = dataframes
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        return "from-mem"

    def supports_cpp_node(self) -> bool:
        return False

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        for i in range(self._repeat_count):
            for k, df in enumerate(self._dataframes):
                x = MessageMeta(df)

                # If we are looping, copy the object. Do this before we push the object in case it changes
                if (i + 1 < self._repeat_count):
                    df = df.copy()

                    # Shift the index to allow for unique indices without reading more data
                    df.index += len(df)
                    self._dataframes[k] = df

                yield x

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node, MessageMeta
