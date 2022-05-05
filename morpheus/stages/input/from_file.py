# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import neo
import typing_utils
from neo.core import operators as ops

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.pipeline.file_types import FileTypes
from morpheus.stages.input.utils import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


class FileSourceStage(SingleOutputSource):
    """
    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be read.
    iterative: boolean
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    file_type : `morpheus.pipeline.file_types.FileTypes`, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null: bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    cudf_kwargs: dict, default=None
        keyword args passed to underlying cuDF I/O function. See the cuDF documentation for `cudf.read_csv()` and
        `cudf.read_json()` for the available options. With `file_type` == 'json', this defaults to ``{ "lines": True }``
        and with `file_type` == 'csv', this defaults to ``{}``.
    """

    def __init__(self,
                 c: Config,
                 filename: str,
                 iterative: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 cudf_kwargs: dict = None):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filename = filename
        self._file_type = file_type
        self._filter_null = filter_null
        self._cudf_kwargs = {} if cudf_kwargs is None else cudf_kwargs

        self._input_count = None
        self._max_concurrent = c.num_threads

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages.
        self._iterative = iterative
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        return "from-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def _build_source(self, seg: neo.Segment) -> StreamPair:

        if CppConfig.get_should_use_cpp():
            out_stream = neos.FileSourceStage(seg, self.unique_name, self._filename, self._repeat_count)
        else:
            out_stream = seg.make_source(self.unique_name, self._generate_frames())

        out_type = MessageMeta

        return out_stream, out_type

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]
        out_type = out_pair[1]

        # Convert our list of dataframes into the desired type. Flatten if necessary
        if (typing_utils.issubtype(out_type, typing.List)):

            def node_fn(input: neo.Observable, output: neo.Subscriber):

                input.pipe(ops.flatten()).subscribe(output)

            flattened = seg.make_node_full(self.unique_name + "-post", node_fn)
            seg.make_edge(out_stream, flattened)
            out_stream = flattened
            out_type = typing.get_args(out_type)[0]

        return super()._post_build_single(seg, (out_stream, out_type))

    def _generate_frames(self):

        df = read_file_to_df(
            self._filename,
            self._file_type,
            filter_nulls=self._filter_null,
            df_type="cudf",
        )

        count = 0

        for _ in range(self._repeat_count):

            x = MessageMeta(df)

            yield x

            count += 1

            # If we are looping, copy and shift the index
            if (self._repeat_count > 0):
                prev_df = df
                df = prev_df.copy()

                df.index += len(df)
