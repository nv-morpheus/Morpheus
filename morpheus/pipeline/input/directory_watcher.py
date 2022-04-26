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

import glob
import logging
import os
import queue

import neo
from watchdog.events import FileSystemEvent
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from watchdog.utils.dirsnapshot import DirectorySnapshot
from watchdog.utils.dirsnapshot import DirectorySnapshotDiff
from watchdog.utils.dirsnapshot import EmptyDirectorySnapshot
from watchdog.utils.patterns import filter_paths

from morpheus._lib.common import FiberQueue
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)


class DirectoryWatcher():
    """
    This class is in responsible of polling for new files in the supplied input glob of directories and 
    forwarding them on to the pipeline for processing.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, `./input_dir/*.json` would read all files with the
        'json' extension in the directory input_dir.
    watch_directory : bool
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    max_files: int
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    sort_glob : bool
        If true the list of files matching `input_glob` will be processed in sorted order.
    """

    def __init__(self, input_glob: str, watch_directory: bool, max_files: int, sort_glob: bool):

        self._input_glob = input_glob
        self._watch_directory = watch_directory
        self._max_files = max_files
        self._sort_glob = sort_glob

        # Determine the directory to watch and the match pattern from the glob
        glob_split = self._input_glob.split("*", 1)

        if (len(glob_split) == 1):
            raise RuntimeError(("When watching directories, input_glob must have a wildcard. "
                                "Otherwise no files will be matched."))

        self._dir_to_watch = os.path.dirname(glob_split[0])
        self._match_pattern = self._input_glob.replace(self._dir_to_watch + "/", "", 1)

        # Will be a watchdog observer if enabled
        self._watcher = None

    def build_node(self, name: str, seg: neo.Segment):

        # The first source just produces filenames
        return seg.make_source(name, self._generate_via_polling())

    def _get_filename_queue(self) -> FiberQueue:
        """
        Returns an async queue with tuples of `([files], is_event)` where `is_event` indicates if this is a file changed
        event (and we should wait for potentially more changes) or if these files were read on startup and should be
        processed immediately.
        """
        q = FiberQueue(128)

        if (self._watch_directory):

            # Create a file watcher
            self._watcher = Observer()
            self._watcher.setDaemon(True)
            self._watcher.setName("DirectoryWatcher")

            event_handler = PatternMatchingEventHandler(patterns=[self._match_pattern])

            def process_dir_change(event: FileSystemEvent):

                # Push files into the queue indicating this is an event
                q.put(([event.src_path], True))

            event_handler.on_created = process_dir_change

            self._watcher.schedule(event_handler, self._dir_to_watch, recursive=True)

            self._watcher.start()

        # Load the glob once and return
        file_list = glob.glob(self._input_glob)
        if self._sort_glob:
            file_list = sorted(file_list)

        if (self._max_files > 0):
            file_list = file_list[:self._max_files]

        logger.info("Found %d files in glob. Loading...", len(file_list))

        # Push all to the queue and close it
        q.put((file_list, False))

        if (not self._watch_directory):
            # Close the queue
            q.close()

        return q

    def _generate_via_polling(self):

        # Its a bit ugly, but utilize a filber queue to yield the thread. This will be improved in the future
        file_queue = FiberQueue(128)

        snapshot = EmptyDirectorySnapshot()

        while (True):

            # Get a new snapshot
            new_snapshot = DirectorySnapshot(self._dir_to_watch, recursive=True)

            # Take the diff from the last one
            diff_events = DirectorySnapshotDiff(snapshot, new_snapshot)
            snapshot = new_snapshot

            files_to_process = filter_paths(diff_events.files_created, included_patterns=[self._input_glob])

            if (self._sort_glob):
                files_to_process = sorted(files_to_process)

            # Convert from generator into list
            files_to_process = list(files_to_process)

            if (len(files_to_process) > 0):
                # is_running = yield files_to_process
                file_queue.put(files_to_process)

            if (not self._watch_directory):
                file_queue.close()

            try:
                batch_timeout = 5.0
                files = file_queue.get(timeout=batch_timeout)

                # We must have gotten a group at startup, process immediately
                if len(files) > 0:
                    yield files

                if (not self._watch_directory):
                    # Break here to prevent looping again
                    break

            except queue.Empty:
                # Timed out, check for files again
                continue

            except Closed:
                # Exit
                break

    def _generate_via_watcher(self):

        # Gets a queue of filenames as they come in. Returns list[str]
        file_queue: FiberQueue = self._get_filename_queue()

        batch_timeout = 5.0

        files_to_process = []

        while True:

            try:
                files, is_event = file_queue.get(timeout=batch_timeout)

                if (is_event):
                    # We may be getting files one at a time from the folder watcher, wait a bit
                    files_to_process = files_to_process + files
                    continue

                # We must have gotten a group at startup, process immediately
                if len(files) > 0:
                    yield files

                # df_queue.task_done()

            except queue.Empty:
                # We timed out, if we have any items in the queue, push those now
                if (len(files_to_process) > 0):
                    yield files_to_process
                    files_to_process = []

            except Closed:
                # Just in case there are any files waiting
                if (len(files_to_process) > 0):
                    yield files_to_process
                    files_to_process = []
                break

