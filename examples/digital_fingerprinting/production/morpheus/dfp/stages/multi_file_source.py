# Copyright (c) 2022, NVIDIA CORPORATION.
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

import fsspec
import fsspec.utils
import pandas as pd
import srf

from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


class MultiFileSource(SingleOutputSource):
    """
    Source stage is used to load messages from files and dumping the contents into the pipeline immediately.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filenames : List[str]
        List of paths to be read from, can be a list of S3 urls (`s3://path`) amd can include wildcard characters `*`
        as defined by `fsspec`:
        https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files
    """

    def __init__(
        self,
        c: Config,
        filenames: typing.List[str],
    ):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filenames = filenames

        self._input_count = None
        self._max_concurrent = c.num_threads

    @property
    def name(self) -> str:
        return "from-multi-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self):
        return False

    def _generate_frames_fsspec(self):

        files: fsspec.core.OpenFiles = fsspec.open_files(self._filenames, filecache={'cache_storage': './.cache/s3tmp'})

        if (len(files) == 0):
            raise RuntimeError(f"No files matched input strings: '{self._filenames}'. "
                               "Check your input pattern and ensure any credentials are correct")

        yield files

    def _build_source(self, builder: srf.Builder) -> StreamPair:

        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes")
        else:
            out_stream = builder.make_source(self.unique_name, self._generate_frames_fsspec())

        return out_stream, fsspec.core.OpenFiles
