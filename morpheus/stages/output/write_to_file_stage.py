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

import os
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd

import cudf

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("to-file", rename_options={"include_index_col": "--include-index-col"})
class WriteToFileStage(SinglePortStage):
    """
    Write all messages to a file.

    This class writes messages to a file.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str
        Name of the file to which the messages will be written.
    overwrite : boolean, default = False, is_flag = True
        Overwrite file if exists. Will generate an error otherwise.
    file_type : `morpheus.common.FileTypes`, optional, case_sensitive = False
        Indicates what type of file to write. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json' and 'jsonlines'
    include_index_col : bool, default = True
        Write out the index as a column, by default True.
    flush : bool, default = False, is_flag = True
        When `True` flush the output buffer to disk on each message.
    """

    def __init__(self,
                 c: Config,
                 filename: str,
                 overwrite: bool = False,
                 file_type: FileTypes = FileTypes.Auto,
                 include_index_col: bool = True,
                 flush: bool = False):

        super().__init__(c)

        self._output_file = filename
        self._overwrite = overwrite

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError("Cannot output classifications to '{}'. File exists and overwrite = False".format(
                    self._output_file))

        self._file_type = file_type

        if (self._file_type == FileTypes.Auto):
            self._file_type = determine_file_type(self._output_file)

        self._is_first = True
        self._include_index_col = include_index_col
        self._flush = flush

    @property
    def name(self) -> str:
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        return True

    def _convert_to_strings(self, df: typing.Union[pd.DataFrame, cudf.DataFrame]):
        if (self._file_type == FileTypes.JSON):
            output_strs = serializers.df_to_json(df, include_index_col=self._include_index_col)
        elif (self._file_type == FileTypes.CSV):
            output_strs = serializers.df_to_csv(df,
                                                include_header=self._is_first,
                                                include_index_col=self._include_index_col)
            self._is_first = False
        else:
            raise NotImplementedError("Unknown file type: {}".format(self._file_type))

        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]

        return output_strs

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Sink to file
        if (self._build_cpp_node()):
            to_file = _stages.WriteToFileStage(builder,
                                               self.unique_name,
                                               self._output_file,
                                               "w",
                                               self._file_type,
                                               self._include_index_col,
                                               self._flush)
        else:

            def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

                # Ensure our directory exists
                os.makedirs(os.path.realpath(os.path.dirname(self._output_file)), exist_ok=True)

                # Open up the file handle
                with open(self._output_file, "a") as out_file:

                    def write_to_file(x: MessageMeta):

                        lines = self._convert_to_strings(x.df)

                        out_file.writelines(lines)

                        if self._flush:
                            out_file.flush()

                        return x

                    obs.pipe(ops.map(write_to_file)).subscribe(sub)

                # File should be closed by here

            to_file = builder.make_node_full(self.unique_name, node_fn)

        builder.make_edge(stream, to_file)
        stream = to_file

        # Return input unchanged to allow passthrough
        return stream, input_stream[1]
