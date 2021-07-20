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

import os
import typing

import typing_utils

from morpheus.config import Config
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair


class WriteToFileStage(SinglePortStage):
    """
    This class writes messages to a file. This class does not buffer or keep the file open between messages.
    It should not be used in production code.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be written
    overwrite : bool
        Overwrite file if exists. Will generate an error otherwise

    """
    def __init__(self, c: Config, filename: str, overwrite: bool):
        super().__init__(c)

        self._output_file = filename
        self._overwrite = overwrite

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError("Cannot output classifications to '{}'. File exists and overwrite = False".format(
                    self._output_file))

    @property
    def name(self) -> str:
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[List[str], ]
            Accepted input types

        """
        return (typing.List[str], )

    def write_to_file(self, x: typing.List[str]):
        """
        Messages are written to a file using this function.

        Parameters
        ----------
        x : typing.List[str]
            Messages that should be written to a file.

        """
        with open(self._output_file, "a") as f:
            f.writelines("\n".join(x))
            f.write("\n")

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Wrap single strings into lists
        if (typing_utils.issubtype(input_stream[1], StreamFuture[str]) or typing_utils.issubtype(input_stream[1], str)):
            stream = stream.map(lambda x: [x])

        # Do a gather just in case we are using dask
        stream = stream.gather()

        # Sink to file
        stream.sink(self.write_to_file)

        # Return input unchanged
        return input_stream
