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

import logging
import typing
from functools import reduce

import mrc
from mrc.core import operators as ops
from tqdm import TMonitor
from tqdm import TqdmSynchronisationWarning
from tqdm import tqdm

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
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


@register_stage("monitor", ignore_args=["determine_count_fn"])
class MonitorStage(SinglePortStage):
    """
    Display throughput numbers at a specific point in the pipeline.

    Monitor stage used to monitor stage performance metrics using Tqdm. Each Monitor Stage will represent one
    line in the console window showing throughput statistics. Can be set up to show an instantaneous
    throughput or average input.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
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
    """
    stage_count: int = 0

    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: float = 0.05,
                 unit: str = "messages",
                 delayed_start: bool = False,
                 determine_count_fn: typing.Callable[[typing.Any], int] = None,
                 log_level: LogLevels = LogLevels.INFO):
        super().__init__(c)

        self._progress: MorpheusTqdm = None
        self._position = MonitorStage.stage_count

        MonitorStage.stage_count += 1

        self._description = description
        self._smoothing = smoothing
        self._unit = unit
        self._delayed_start = delayed_start

        self._determine_count_fn = determine_count_fn

        self._log_level = log_level
        self._enabled = None  # defined on first call to _is_enabled

    @property
    def name(self) -> str:
        return "monitor"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _is_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = logger.isEnabledFor(self._log_level)

        return self._enabled

    def on_start(self):
        """
        Starts the pipeline stage's progress bar.
        """
        if self._is_enabled():
            # Set the monitor interval to 0 to use prevent using tqdms monitor
            tqdm.monitor_interval = 0

            # Start the progress bar if we dont have a delayed start
            if (not self._delayed_start):
                self._ensure_progress_bar()

    def stop(self):
        """
        Clean up and close the progress bar.
        """
        if (self._progress is not None):
            self._progress.close()

    def _ensure_progress_bar(self):
        if (self._progress is None):
            self._progress = MorpheusTqdm(desc=self._description,
                                          smoothing=self._smoothing,
                                          dynamic_ncols=True,
                                          unit=(self._unit if self._unit.startswith(" ") else " {}".format(self._unit)),
                                          mininterval=0.25,
                                          maxinterval=1.0,
                                          miniters=1,
                                          position=self._position)

            self._progress.reset()

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        if not self._is_enabled():
            return input_stream

        def sink_on_completed():
            # Set the name to complete. This refreshes the display
            self._progress.set_description_str(self._progress.desc + "[Complete]")

            self._progress.stop()

            # To prevent the monitors from writing over eachother, stop the monitor when the last stage completes
            MonitorStage.stage_count -= 1

            if (MonitorStage.stage_count <= 0 and MorpheusTqdm.monitor is not None):
                MorpheusTqdm.monitor.exit()
                MorpheusTqdm.monitor = None

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            obs.pipe(ops.map(self._progress_sink), ops.on_completed(sink_on_completed)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)

        builder.make_edge(input_stream[0], stream)

        return stream, input_stream[1]

    def _refresh_progress(self, _):
        self._progress.refresh()

    def _progress_sink(self, x):

        # Make sure the progress bar is shown
        self._ensure_progress_bar()

        if (self._determine_count_fn is None):
            self._determine_count_fn = self._auto_count_fn(x)

        # Skip incase we have empty objects
        if (self._determine_count_fn is None):
            return x

        # Do our best to determine the count
        n = self._determine_count_fn(x)

        self._progress.update(n=n)

        return x

    def _auto_count_fn(self, x):

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
        elif (isinstance(x, list)):
            item_count_fn = self._auto_count_fn(x[0])
            return lambda y: reduce(lambda sum, z, item_count_fn=item_count_fn: sum + item_count_fn(z), y, 0)
        elif (isinstance(x, str)):
            return lambda y: 1
        elif (hasattr(x, "__len__")):
            return len  # Return len directly (same as `lambda y: len(y)`)
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(x)))
