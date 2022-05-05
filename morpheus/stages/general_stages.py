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
from functools import reduce

import cupy as cp
import neo
from neo.core import operators as ops
from tqdm import TMonitor
from tqdm import TqdmSynchronisationWarning
from tqdm import tqdm

import cudf

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.pipeline import Stage
from morpheus.messages.messages import MessageMeta
from morpheus.messages.messages import MultiMessage
from morpheus.messages.messages import MultiResponseProbsMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.logging import deprecated_stage_warning

logger = logging.getLogger(__name__)


class BufferStage(SinglePortStage):
    """
    The input messages are buffered by this stage class for faster access to downstream stages. Allows
    upstream stages to run faster than downstream stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config, count: int = 1000):
        super().__init__(c)

        self._buffer_count = count

    @property
    def name(self) -> str:
        return "buffer"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # This stage is no longer needed and is just a pass thru stage
        deprecated_stage_warning(logger, type(self), self.unique_name)

        return input_stream


class DelayStage(SinglePortStage):
    """
    Delay stage class. Used to buffer all inputs until the timeout duration is hit. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config, duration: str):
        super().__init__(c)

        self._duration = duration

    @property
    def name(self) -> str:
        return "delay"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # This stage is no longer needed and is just a pass thru stage
        deprecated_stage_warning(logger, type(self), self.unique_name)

        return input_stream


class TriggerStage(SinglePortStage):
    """
    This stage will buffer all inputs until the source stage is complete. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "trigger"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # Store all messages until on_complete is called and then push them
        def node_fn(input: neo.Observable, output: neo.Subscriber):

            input.pipe(ops.to_list(), ops.flatten()).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(input_stream[0], node)

        return node, input_stream[1]


# Functions exactly the same as TMonitor, except we do not check for `instance.miniters == 1` before updating. This
# allows the timer to update every 1 second on the screen making the pipeline feel running
class MorpheusTqdmMonitor(TMonitor):

    def run(self):
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

        self.last_update_t = self._time()

        return super().update(n)

    def stop(self):

        # Set is running to false to stop elapsed from incrementing
        self.is_running = False


class MonitorStage(SinglePortStage):
    """
    Monitor stage used to monitor stage performance metrics using Tqdm. Each Monitor Stage will represent one
    line in the console window showing throughput statistics. Can be set up to show an instantaneous
    throughput or average input.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    description : str
        Name to show for this Monitor Stage in the console window.
    smoothing : float
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    delayed_start : bool
        Delay start of progress bar.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.

    """
    stage_count: int = 0

    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: float = 0.05,
                 unit="messages",
                 delayed_start: bool = False,
                 determine_count_fn: typing.Callable[[typing.Any], int] = None):
        super().__init__(c)

        self._progress: MorpheusTqdm = None
        self._position = MonitorStage.stage_count

        MonitorStage.stage_count += 1

        self._description = description
        self._smoothing = smoothing
        self._unit = unit
        self._delayed_start = delayed_start

        self._determine_count_fn = determine_count_fn

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

    def on_start(self):

        # Set the monitor interval to 0 to use prevent using tqdms monitor
        tqdm.monitor_interval = 0

        # Start the progress bar if we dont have a delayed start
        if (not self._delayed_start):
            self._ensure_progress_bar()

    def stop(self):
        if (self._progress is not None):
            self._progress.close()

    def _ensure_progress_bar(self):
        if (self._progress is None):
            self._progress = MorpheusTqdm(desc=self._description,
                                          smoothing=self._smoothing,
                                          dynamic_ncols=True,
                                          unit=self._unit,
                                          mininterval=0.25,
                                          maxinterval=1.0,
                                          miniters=1,
                                          position=self._position)

            self._progress.reset()

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        def sink_on_error(x):
            logger.error("Node: '%s' received error: %s", self.unique_name, x)

        def sink_on_completed():
            # Set the name to complete. This refreshes the display
            self._progress.set_description_str(self._progress.desc + "[Complete]")

            self._progress.stop()

            # To prevent the monitors from writing over eachother, stop the monitor when the last stage completes
            MonitorStage.stage_count -= 1

            if (MonitorStage.stage_count <= 0 and MorpheusTqdm.monitor is not None):
                MorpheusTqdm.monitor.exit()
                MorpheusTqdm.monitor = None

        stream = seg.make_sink(self.unique_name, self._progress_sink, sink_on_error, sink_on_completed)

        seg.make_edge(input_stream[0], stream)

        return input_stream

    def _refresh_progress(self, _):
        self._progress.refresh()

    def _progress_sink(self, x):

        # Make sure the progress bar is shown
        self._ensure_progress_bar()

        if (self._determine_count_fn is None):
            self._determine_count_fn = self._auto_count_fn(x)

        # Skip incase we have empty objects
        if (self._determine_count_fn is None):
            return

        # Do our best to determine the count
        n = self._determine_count_fn(x)

        self._progress.update(n=n)

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


class AddClassificationsStage(SinglePortStage):
    """
    Add classification labels based on probabilities calculated in inference stage. Label indexes will be looked up in
    the Config.class_labels property. Uses default threshold of 0.5 for predictions.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    threshold : float
        Threshold to classify, default is 0.5.
    labels: list, default = None
        The list of labels to add classifications for. Each item in the list will determine its index from the
        Config.class_labels property and must be one of the available class labels. Leave as None to add all labels in
        the Config.class_labels property.
    prefix: str, default = ""
        A prefix to append to each label.

    """

    def __init__(self, c: Config, threshold: float = 0.5, labels: typing.List[str] = None, prefix: str = ""):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._threshold = threshold
        self._prefix = prefix
        self._class_labels = c.class_labels
        self._labels = labels if labels is not None and len(labels) > 0 else c.class_labels

        # Build the Index to Label map.
        self._idx2label = {}

        for label in self._labels:
            # All labels must be in class_labels in order to get their position
            if (label not in self._class_labels):
                logger.warning("The label '%s' is not in Config.class_labels and will be ignored", label)
                continue

            self._idx2label[self._class_labels.index(label)] = self._prefix + label

        assert len(self._idx2label) > 0, "No labels were added to the stage"

    @property
    def name(self) -> str:
        return "add-class"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return True

    def _add_labels(self, x: MultiResponseProbsMessage):

        if (x.probs.shape[1] != len(self._class_labels)):
            raise RuntimeError("Label count does not match output of model. Label count: {}, Model output: {}".format(
                len(self._class_labels), x.probs.shape[1]))

        probs_np = (x.probs > self._threshold).astype(bool).get()

        for i, label in self._idx2label.items():
            x.set_meta(label, probs_np[:, i].tolist())

        # Return passthrough
        return x

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        if CppConfig.get_should_use_cpp():
            stream = neos.AddClassificationsStage(seg,
                                                  self.unique_name,
                                                  self._threshold,
                                                  len(self._class_labels),
                                                  self._idx2label)
        else:
            stream = seg.make_node(self.unique_name, self._add_labels)

        seg.make_edge(input_stream[0], stream)

        # Return input unchanged
        return stream, MultiResponseProbsMessage


class FilterDetectionsStage(SinglePortStage):
    """
    This Stage class is used to filter results based on a given criteria.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    threshold : float
        Threshold to classify, default is 0.5.

    """

    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "filter"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return True

    def filter(self, x: MultiResponseProbsMessage) -> typing.List[MultiResponseProbsMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiResponseProbsMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        typing.List[`morpheus.pipeline.messages.MultiResponseProbsMessage`]
            List of filtered messages.

        """
        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        output_list = []

        # Get per row detections
        detections = (x.probs > self._threshold).any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = cp.concatenate([cp.array([False]), detections, cp.array([False])])

        true_pairs = cp.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

        for pair in true_pairs:
            pair = tuple(pair.tolist())
            mess_offset = x.mess_offset + pair[0]
            mess_count = pair[1] - pair[0]

            # Filter empty message groups
            if (mess_count == 0):
                continue

            output_list.append(
                MultiResponseProbsMessage(x.meta,
                                          mess_offset=mess_offset,
                                          mess_count=mess_count,
                                          memory=x.memory,
                                          offset=pair[0],
                                          count=mess_count))

        return output_list

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # Convert list back to single MultiResponseProbsMessage
        def flatten_fn(input: neo.Observable, output: neo.Subscriber):

            input.pipe(ops.map(self.filter), ops.flatten()).subscribe(output)

        if CppConfig.get_should_use_cpp():
            stream = neos.FilterDetectionsStage(seg, self.unique_name, self._threshold)
        else:
            stream = seg.make_node_full(self.unique_name, flatten_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage


class ZipStage(Stage):

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "zip"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        raise NotImplementedError(("The ZipStage has been deprecated and is not longer supported. "
                                   "Non-linear pipelines will be added in a future release"))


class MergeStage(Stage):

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "merge"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        raise NotImplementedError(("The MergeStage has been deprecated and is not longer supported. "
                                   "Non-linear pipelines will be added in a future release"))


class SwitchStage(Stage):

    def __init__(self, c: Config, num_outputs: int, predicate: typing.Callable[[typing.Any], int]):
        super().__init__(c)

        self._num_outputs = num_outputs
        self._predicate = predicate

        self._create_ports(1, num_outputs)

    @property
    def name(self) -> str:
        return "sample"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        raise NotImplementedError(("The SwitchStage has been deprecated and is not longer supported. "
                                   "Non-linear pipelines will be added in a future release"))


class AddScoresStage(SinglePortStage):
    """
    Add score labels based on probabilities calculated in inference stage. Label indexes will be looked up in
    the Config.class_labels property.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    labels: list, default = None
        The list of labels to add classifications for. Each item in the list will determine its index from the
        Config.class_labels property and must be one of the available class labels. Leave as None to add all labels in
        the Config.class_labels property.
    prefix: str, default = ""
        A prefix to append to each label.

    """

    def __init__(self, c: Config, labels: typing.List[str] = None, prefix: str = ""):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._prefix = prefix
        self._class_labels = c.class_labels
        self._labels = labels if labels is not None and len(labels) > 0 else c.class_labels

        # Build the Index to Label map.
        self._idx2label = {}

        for label in self._labels:
            # All labels must be in class_labels in order to get their position
            if (label not in self._class_labels):
                logger.warning("The label '%s' is not in Config.class_labels and will be ignored", label)
                continue

            self._idx2label[self._class_labels.index(label)] = self._prefix + label

        assert len(self._idx2label) > 0, "No labels were added to the stage"

    @property
    def name(self) -> str:
        return "add-scores"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    @classmethod
    def supports_cpp_node(cls):
        # Enable support by default
        return True

    def _add_labels(self, x: MultiResponseProbsMessage):

        if (x.probs.shape[1] != len(self._class_labels)):
            raise RuntimeError("Label count does not match output of model. Label count: {}, Model output: {}".format(
                len(self._class_labels), x.probs.shape[1]))

        probs_np = x.probs.get()

        for i, label in self._idx2label.items():
            x.set_meta(label, probs_np[:, i].tolist())

        # Return passthrough
        return x

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        if CppConfig.get_should_use_cpp():
            stream = neos.AddScoresStage(seg, self.unique_name, len(self._class_labels), self._idx2label)
        else:
            stream = seg.make_node(self.unique_name, self._add_labels)

        seg.make_edge(input_stream[0], stream)

        # Return input unchanged
        return stream, input_stream[1]
