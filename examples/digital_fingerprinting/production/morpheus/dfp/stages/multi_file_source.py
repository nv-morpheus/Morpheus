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
"""Source stages for reading multiple files from a list of fsspec paths."""

import logging
import time
import typing

import fsspec
import fsspec.utils
import mrc

from morpheus.config import Config
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(f"morpheus.{__name__}")


class MultiFileSource(SingleOutputSource):
    """
    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filenames : List[str]
        List of paths to be read from, can be a list of S3 urls (`s3://path`) amd can include wildcard characters `*`
        as defined by `fsspec`:
        https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files
    watch : bool, default = False
        When True, will check `filenames` for new files and emit them as they appear. This assumes that at least one of
        the paths in `filenames` contains a wildecard character.
    watch_interval : float, default = 1.0
        When `watch` is True, this is the time in seconds between polling the paths in `filenames` for new files.
        Ignored when `watch` is False.
    """

    def __init__(
        self,
        c: Config,
        filenames: typing.List[str],
        watch: bool = False,
        watch_interval: float = 1.0,
    ):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._filenames = filenames

        self._input_count = None
        self._max_concurrent = c.num_threads
        self._watch = watch
        self._watch_interval = watch_interval

    @property
    def name(self) -> str:
        """Return the name of the stage."""
        return "from-multi-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self):
        """Indicates whether this stage supports C++ nodes."""
        return False

    def _generate_frames_fsspec(self) -> typing.Iterable[fsspec.core.OpenFiles]:

        files: fsspec.core.OpenFiles = fsspec.open_files(self._filenames)

        if (len(files) == 0):
            raise RuntimeError(f"No files matched input strings: '{self._filenames}'. "
                               "Check your input pattern and ensure any credentials are correct")

        yield files

    def _polling_generate_frames_fsspec(self) -> typing.Iterable[fsspec.core.OpenFiles]:
        files_seen = set()
        curr_time = time.monotonic()
        next_update_epoch = curr_time

        while (True):
            # Before doing any work, find the next update epoch after the current time
            while (next_update_epoch <= curr_time):
                # Only ever add `self._watch_interval` to next_update_epoch so all updates are at repeating intervals
                next_update_epoch += self._watch_interval

            file_set = set()
            filtered_files = []

            files = fsspec.open_files(self._filenames)
            for file in files:
                file_set.add(file.full_name)
                if file.full_name not in files_seen:
                    filtered_files.append(file)

            # Replace files_seen with the new set of files. This prevents a memory leak that could occurr if files are
            # deleted from the input directory. In addition if a file with a given name was created, seen/processed by
            # the stage, and then deleted, and a new file with the same name appeared sometime later, the stage will
            # need to re-ingest that new file.
            files_seen = file_set

            if len(filtered_files) > 0:
                yield fsspec.core.OpenFiles(filtered_files, fs=files.fs)

            curr_time = time.monotonic()

            # If we spent more than `self._watch_interval` doing work and/or yielding to the output channel blocked,
            # then we should only sleep for the remaining time until the next update epoch.
            sleep_duration = next_update_epoch - curr_time
            if (sleep_duration > 0):
                time.sleep(sleep_duration)
                curr_time = time.monotonic()

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes")

        if self._watch:
            out_stream = builder.make_source(self.unique_name, self._polling_generate_frames_fsspec())
        else:
            out_stream = builder.make_source(self.unique_name, self._generate_frames_fsspec())

        return out_stream, fsspec.core.OpenFiles
