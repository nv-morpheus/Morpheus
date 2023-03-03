# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import json
import logging
import typing

import fsspec
import fsspec.utils
import mrc

from morpheus.config import Config
from morpheus.messages.message_control import MessageControl
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger("morpheus.{}".format(__name__))


class ControlMessageSourceStage(SingleOutputSource):
    """
    Source stage is used to recieve control messages from different sources.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filenames : List[str]
        List of paths to be read from, can be a list of S3 urls (`s3://path`) amd can include wildcard characters `*`
        as defined by `fsspec`:
        https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files
    """

    def __init__(self, c: Config, filenames: typing.List[str]):
        super().__init__(c)
        self._filenames = filenames

    @property
    def name(self) -> str:
        return "from-message-control"

    def supports_cpp_node(self):
        return True

    def _create_control_message(self) -> MessageControl:

        openfiles: fsspec.core.OpenFiles = fsspec.open_files(self._filenames)

        if (len(openfiles) == 0):
            raise RuntimeError(f"No files matched input strings: '{self._filenames}'. "
                               "Check your input pattern and ensure any credentials are correct")

        # TODO(Devin): Support multiple tasks in a single file
        for openfile in openfiles:
            with openfile as f:
                message_configs = json.load(f)
                for message_config in message_configs.get("inputs", []):
                    message_control = MessageControl(message_config)
                    yield message_control

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        out_stream = builder.make_source(self.unique_name, self._create_control_message())

        return out_stream, fsspec.core.OpenFiles
