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

import logging
import typing
from functools import reduce

from tqdm import TMonitor
from tqdm import TqdmSynchronisationWarning
from tqdm import tqdm

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.utils.logger import LogLevels

logger = logging.getLogger(__name__)


class MorpheusTqdmMonitor(TMonitor):
    """
    Monitoring thread for tqdm bars.
    """
    def run(self):
        """
        This function does exactly the same as `TMonitor.run`, except we do not check for `instance.miniters == 1`
        before updating. This allows the timer to update every 1 second on the screen making the pipeline feel running
        """
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                # Check tqdm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if ((cur_t - instance.last_print_t) >= instance.maxinterval):
                        # Refresh now! (works only for manual tqdm)
                        instance.refresh(nolock=True)
                    # Remove accidental long-lived strong reference
                    del instance
                if instances != self.get_instances():  # pragma: nocover
                    logging.warn("Set changed size during iteration" + " (see https://github.com/tqdm/tqdm/issues/481)",
                                 TqdmSynchronisationWarning,
                                 stacklevel=2)
                # Remove accidental long-lived strong references
                del instances


class MorpheusTqdm(tqdm):
    """
    Subclass of tqdm to provide slightly different functionality with their TMonitor.

    """
    monitor_interval = 1  # set to 0 to disable the thread
    monitor: MorpheusTqdmMonitor = None

    def __new__(cls, *args, **kwargs):
        with cls.get_lock():  # also constructs lock if non-existent

            if (cls.monitor is None or not cls.monitor.report()):
                # Set the new type of monitor
                cls.monitor = MorpheusTqdmMonitor(cls, cls.monitor_interval)

        return tqdm.__new__(cls, args, kwargs)

    def __init__(self, *args, **kwargs):

        # Must set this first
        self.is_running = True

        super().__init__(*args, **kwargs)

        self.last_update_t = self.start_t

    @property
    def format_dict(self):

        base_val = super().format_dict

        # If we arent running, dont increment the time
        if (not self.is_running):
            base_val["elapsed"] = self.last_update_t - self.start_t

        return base_val

    def update(self, n=1):
        """
        This function updates the time and progress bar.

        Parameters
        ----------
        n : int
            Increment to add to the internal counter of iterations.
        """

        self.last_update_t = self._time()

        return super().update(n)

    def stop(self):
        """
        Progress bar incrementing is stopped by this function.
        """
        # Set is running to false to stop elapsed from incrementing
        self.is_running = False


class SilentMorpheusTqdm(MorpheusTqdm):
    """
    Subclass of MorpheusTqdm to silent monitors, it provides slightly different functionality with their TMonitor.

    """
    def refresh(self, nolock=False, lock_args=None):
        return


class MonitorController:
    """
    Controls and displays throughput numbers at a specific point in the pipeline.

    Parameters
    ----------
    position: int
        Specifies the monitor's position on the console.
    description : str, default = "Progress"
        Name to show for this Monitor Stage in the console window.
    smoothing : float
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    delayed_start : bool
        When delayed_start is enabled, the progress bar will not be shown until the first message is received.
        Otherwise, the progress bar is shown on pipeline startup and will begin timing immediately. In large pipelines,
        this option may be desired to give a more accurate timing.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.
    log_level : `morpheus.utils.logger.LogLevels`, default = 'INFO'
        Enable this stage when the configured log level is at `log_level` or lower.
    tqdm_class: `tqdm`, default = None
        Custom implementation of tqdm if required.
    """

    controller_count: int = 0

    def __init__(self,
                 position: int,
                 description: str,
                 smoothing: float,
                 unit: str,
                 delayed_start: bool,
                 determine_count_fn: typing.Callable[[typing.Any], int],
                 log_level: LogLevels,
                 tqdm_class: tqdm = None):

        self._progress: tqdm = None
        self._position = position
        self._description = description
        self._smoothing = smoothing
        self._unit = unit
        self._delayed_start = delayed_start
        self._determine_count_fn = determine_count_fn
        self._tqdm_class = tqdm_class if tqdm_class else MorpheusTqdm

        if isinstance(log_level, LogLevels):
            log_level = log_level.value

        self._log_level = log_level
        self._enabled = None  # defined on first call to _is_enabled

    @property
    def delayed_start(self) -> bool:
        return self._delayed_start

    @property
    def progress(self) -> tqdm:
        return self._progress

    def is_enabled(self) -> bool:
        """
        Returns a boolean indicating whether or not the logger is enabled.
        """

        if self._enabled is None:
            self._enabled = logger.isEnabledFor(self._log_level)

        return self._enabled

    def ensure_progress_bar(self):
        """
        Ensures that the progress bar is initialized and ready for display.
        """

        if (self._progress is None):
            self._progress = self._tqdm_class(desc=self._description,
                                              smoothing=self._smoothing,
                                              dynamic_ncols=True,
                                              unit=(self._unit if self._unit.startswith(" ") else f" {self._unit}"),
                                              mininterval=0.25,
                                              maxinterval=1.0,
                                              miniters=1,
                                              position=self._position)

            self._progress.reset()

    def refresh_progress(self, _):
        """
        Refreshes the progress bar display.
        """
        self._progress.refresh()

    def progress_sink(self, x: typing.Union[cudf.DataFrame, MultiMessage, MessageMeta, ControlMessage, typing.List]):
        """
        Receives a message and determines the count of the message.
        The progress bar is displayed and the progress is updated.

        Parameters
        ----------
        x: typing.Union[cudf.DataFrame, MultiMessage, MessageMeta, ControlMessage, typing.List]
            Message that determines the count of the message

        Returns
        -------
        x: typing.Union[cudf.DataFrame, MultiMessage, MessageMeta, ControlMessage, typing.List]

        """

        # Make sure the progress bar is shown
        self.ensure_progress_bar()

        if (self._determine_count_fn is None):
            self._determine_count_fn = self.auto_count_fn(x)

        # Skip incase we have empty objects
        if (self._determine_count_fn is None):
            return x

        # Do our best to determine the count
        n = self._determine_count_fn(x)

        self._progress.update(n=n)

        return x

    def auto_count_fn(self, x: typing.Union[cudf.DataFrame, MultiMessage, MessageMeta, ControlMessage, typing.List]):
        """
        This is a helper function that is used to determine the count of messages received by the
        monitor.

        Parameters
        ----------
        x: typing.Union[cudf.DataFrame, MultiMessage, MessageMeta, ControlMessage, typing.List]
            Message that determines the count of the message

        Returns
        -------
        Message count.

        """

        if (x is None):
            return None

        # Wait for a list thats not empty
        if (isinstance(x, list) and len(x) == 0):
            return None

        if (isinstance(x, cudf.DataFrame)):
            return lambda y: len(y.index)
        elif (isinstance(x, MultiMessage)):
            return lambda y: y.mess_count
        elif (isinstance(x, MessageMeta)):
            return lambda y: y.count
        elif isinstance(x, ControlMessage):

            def check_df(y):
                df = y.payload().df
                if df is not None:
                    return len(df)
                else:
                    return 0

            return check_df
        elif (isinstance(x, list)):
            item_count_fn = self.auto_count_fn(x[0])
            return lambda y: reduce(lambda sum, z, item_count_fn=item_count_fn: sum + item_count_fn(z), y, 0)
        elif (isinstance(x, str)):
            return lambda y: 1
        elif (hasattr(x, "__len__")):
            return len  # Return len directly (same as `lambda y: len(y)`)
        else:
            return lambda y: 1

    def sink_on_completed(self):
        """
        Stops the progress bar and prevents the monitors from writing over each other when the last
        stage completes.
        """

        # Set the name to complete. This refreshes the display
        self.progress.set_description_str(self.progress.desc + "[Complete]")

        self.progress.stop()

        # To prevent the monitors from writing over eachother, stop the monitor when the last stage completes
        MonitorController.controller_count -= 1

        if (MonitorController.controller_count <= 0 and self._tqdm_class.monitor is not None):
            self._tqdm_class.monitor.exit()
            self._tqdm_class.monitor = None
