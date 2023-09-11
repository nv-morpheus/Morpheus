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
"""File source stage."""

import logging
import time
import typing
from functools import partial
from urllib.parse import urlsplit

import fsspec
import mrc
from mrc.core import operators as ops

from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.directory_watcher import DirectoryWatcher

logger = logging.getLogger(__name__)


@register_stage("file-source", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class FileSource(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    files : List[str]
        List of paths to be read from, can be a list of S3 URLs (`s3://path`) and can include wildcard characters `*`
        as defined by `fsspec`:
        https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files
    watch : bool, default = False
        When True, will check `files` for new files and emit them as they appear.  (Note: `watch_interval` is
        applicable when `watch` is True and there are no remote paths in `files`.)
    watch_interval : float, default = 1.0
        When `watch` is True, this is the time in seconds between polling the paths in `files` for new files.
        (Note: Applicable when path in `files` are remote and when `watch` is True)
    sort_glob : bool, default = False
        If true, the list of files matching `input_glob` will be processed in sorted order.
        (Note: Applicable when all paths in `files` are local.)
    recursive : bool, default = True
        If true, events will be emitted for the files in subdirectories matching `input_glob`.
        (Note: Applicable when all paths in `files` are local.)
    queue_max_size : int, default = 128
        Maximum queue size to hold the file paths to be processed that match `input_glob`.
        (Note: Applicable when all paths in `files` are local.)
    batch_timeout : float, default = 5.0
        Timeout to retrieve batch messages from the queue.
        (Note: Applicable when all paths in `files` are local.)
    file_type : `morpheus.common.FileTypes`, optional, case_sensitive = False
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'csv', 'json', 'jsonlines' and 'parquet'.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null : bool, default = True
        Whether or not to filter rows with a null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    parser_kwargs : dict, default = {}
        Extra options to pass to the file parser.
    """

    def __init__(self,
                 c: Config,
                 files: typing.List[str],
                 watch: bool = False,
                 watch_interval: float = 1.0,
                 sort_glob: bool = False,
                 recursive: bool = True,
                 queue_max_size: int = 128,
                 batch_timeout: float = 5.0,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 parser_kwargs: dict = None):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        if not files:
            raise ValueError("The 'files' cannot be empty.")

        if watch and len(files) != 1:
            raise ValueError("When 'watch' is True, the 'files' should contain exactly one file path.")

        self._files = list(files)
        self._watch = watch
        self._sort_glob = sort_glob
        self._recursive = recursive
        self._queue_max_size = queue_max_size
        self._batch_timeout = batch_timeout
        self._file_type = file_type
        self._filter_null = filter_null
        self._parser_kwargs = parser_kwargs or {}
        self._watch_interval = watch_interval
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "file-source"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return True

    def _has_remote_paths(self):
        return any(urlsplit(file).scheme for file in self._files if "://" in file)

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes")

        if self._watch and not self._has_remote_paths():
            # When watching a directory, we use the directory path for monitoring.
            input_glob = self._files[0]
            watcher = DirectoryWatcher(input_glob=input_glob,
                                       watch_directory=self._watch,
                                       max_files=None,
                                       sort_glob=self._sort_glob,
                                       recursive=self._recursive,
                                       queue_max_size=self._queue_max_size,
                                       batch_timeout=self._batch_timeout)
            out_stream = watcher.build_node(self.unique_name, builder)

            out_type = typing.List[str]
        else:
            if self._watch:
                generator_function = self._polling_generate_frames_fsspec
            else:
                generator_function = self._generate_frames_fsspec

            out_stream = builder.make_source(self.unique_name, generator_function())
            out_type = fsspec.core.OpenFiles

        # Supposed to just return a source here
        return out_stream, out_type

    def _generate_frames_fsspec(self) -> typing.Iterable[fsspec.core.OpenFiles]:

        files: fsspec.core.OpenFiles = fsspec.open_files(self._files)

        if (len(files) == 0):
            raise RuntimeError(f"No files matched input strings: '{self._files}'. "
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

            files = fsspec.open_files(self._files)
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

    @staticmethod
    def generate_frames(file: fsspec.core.OpenFiles,
                        file_type: FileTypes,
                        filter_null: bool,
                        parser_kwargs: dict,
                        repeat_count: int) -> list[MessageMeta]:
        """
        Generate message frames from a file.

        This function reads data from a file and generates message frames (MessageMeta) based on the file's content.
        It can be used to load and process messages from a file for testing and analysis within a Morpheus pipeline.

        Parameters
        ----------
        file : fsspec.core.OpenFiles
            An open file object obtained using fsspec's `open_files` function.
        file_type : FileTypes
            Indicates the type of the file to read. Supported types include 'csv', 'json', 'jsonlines', and 'parquet'.
        filter_null : bool
            Determines whether to filter out rows with null values in the 'data' column. Filtering null values is
            recommended to prevent potential issues during processing.
        parser_kwargs : dict
            Additional keyword arguments to pass to the file parser.
        repeat_count : int
            The number of times to repeat the data reading process. Each repetition generates a new set of message
            frames.

        Returns
        -------
        List[MessageMeta]
            MessageMeta objects, each containing a dataframe of messages from the file.
        """
        df = read_file_to_df(
            file.full_name,
            file_type=file_type,
            filter_nulls=filter_null,
            parser_kwargs=parser_kwargs,
            df_type="cudf",
        )

        metas = []

        for i in range(repeat_count):

            x = MessageMeta(df)

            # If we are looping, copy the object. Do this before we push the object in case it changes
            if (i + 1 < repeat_count):
                df = df.copy()

                # Shift the index to allow for unique indices without reading more data
                df.index += len(df)

            metas.append(x)

        return metas

    @staticmethod
    def convert_list_to_fsspec_files(
            files: typing.Union[typing.List[str], fsspec.core.OpenFiles]) -> fsspec.core.OpenFiles:
        """
        Convert a list of file paths to fsspec OpenFiles.

        This static method takes a list of file paths or an existing fsspec OpenFiles object and ensures that the
        input is converted to an OpenFiles object for uniform handling in Morpheus pipeline stages.

        Parameters
        ----------
        files : Union[List[str], fsspec.core.OpenFiles]
            A list of file paths or an existing fsspec OpenFiles object.

        Returns
        -------
        fsspec.core.OpenFiles
            An fsspec OpenFiles object representing the input files.
        """

        if isinstance(files, list):
            files: fsspec.core.OpenFiles = fsspec.open_files(files)

        return files

    def _post_build_single(self, builder: mrc.Builder, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]

        post_node = builder.make_node(
            self.unique_name + "-post",
            ops.map(self.convert_list_to_fsspec_files),
            ops.flatten(),
            ops.map(
                partial(self.generate_frames,
                        file_type=self._file_type,
                        filter_null=self._filter_null,
                        parser_kwargs=self._parser_kwargs,
                        repeat_count=self._repeat_count)),
            ops.flatten())

        builder.make_edge(out_stream, post_node)

        out_stream = post_node
        out_type = MessageMeta

        return super()._post_build_single(builder, (out_stream, out_type))
