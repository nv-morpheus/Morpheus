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

import typing
from functools import reduce

import cupy as cp
import streamz
from streamz.core import Stream
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import unpack_tuple
from morpheus.utils.type_utils import unpack_union


class BufferStage(SinglePortStage):
    """
    The input messages are buffered by this stage class for faster access to downstream stages. Allows
    upstream stages to run faster than downstream stages.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

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
            Accepted input types

        """
        return (typing.Any, )

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        return input_stream[0].buffer(self._buffer_count), input_stream[1]


class DelayStage(SinglePortStage):
    """
    Delay stage class. Used to buffer all inputs until the timeout duration is hit. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

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
            Accepted input types

        """
        return (typing.Any, )

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        return input_stream[0].time_delay(self._duration), input_stream[1]


class TriggerStage(SinglePortStage):
    """
    This stage will buffer all inputs until the source stage is complete. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

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
            Accepted input types

        """
        return (typing.Any, )

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        collector = stream.collect()

        def flush_input(_: Stream):
            collector.flush()

        stream.add_done_callback(flush_input)

        stream = collector.flatten()

        return stream, input_stream[1]


class MonitorStage(SinglePortStage):
    """
    Monitor stage used to monitor stage performance metrics using Tqdm. Each Monitor Stage will represent one
    line in the console window showing throughput statistics. Can be set up to show an instantaneous
    throughput or average input.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    description : str
        Name to show for this Monitor Stage in the console window
    smoothing : int
        Smoothing parameter to determine how much the throughput should be averaged. 0 = Instantaneous, 1 =
        Average.
    unit : str
        Units to show in the rate value.
    determine_count_fn : typing.Callable[[typing.Any], int]
        Custom function for determining the count in a message. Gets called for each message. Allows for
        correct counting of batched and sliced messages.

    """
    def __init__(self,
                 c: Config,
                 description: str = "Progress",
                 smoothing: int = 0.05,
                 unit="messages",
                 determine_count_fn: typing.Callable[[typing.Any], int] = None):
        super().__init__(c)

        self._progress: tqdm = None

        self._description = description
        self._smoothing = smoothing
        self._unit = unit

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
            Accepted input types

        """
        return (typing.Any, )

    def on_start(self):

        self._progress = tqdm(desc=self._description,
                              smoothing=self._smoothing,
                              dynamic_ncols=True,
                              unit=self._unit,
                              mininterval=0.25,
                              maxinterval=1.0)

        self._progress.reset()

    async def stop(self):
        if (self._progress is not None):
            self._progress.close()

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        # Add the progress sink to the current stream. Use a gather here just in case its a Dask stream
        input_stream[0].gather().sink(self._progress_sink)

        return input_stream

    def _refresh_progress(self, _):
        self._progress.refresh()

    def _progress_sink(self, x):

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
        elif (isinstance(x, list)):
            item_count_fn = self._auto_count_fn(x[0])
            sum_fn = lambda sum, z: sum + item_count_fn(z)
            return lambda y: reduce(sum_fn, y, 0)
        elif (isinstance(x, str)):
            return lambda y: 1
        elif (hasattr(x, "__len__")):
            return len  # Return len directly (same as `lambda y: len(y)`)
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(x)))


class AddClassificationsStage(SinglePortStage):
    """
    Add classification labels based on probabilities calculated in inference stage. Uses default threshold of
    0.5 for predictions.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    threshold : float
        Threshold to classify, default is 0.5

    """
    def __init__(self,
                 c: Config,
                 threshold: float = 0.5,
                 labels_file: str = None,
                 labels: typing.List[str] = None,
                 prefix: str = ""):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._threshold = threshold
        self._labels_file = labels_file
        self._labels = labels
        self._prefix = prefix

    @property
    def name(self) -> str:
        return "add-class"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseProbsMessage, ]
            Accepted input types

        """
        return (MultiResponseProbsMessage, )

    def _determine_labels(self) -> typing.List[str]:

        if (self._labels is not None):
            return self._labels
        elif (self._labels_file is not None):
            raise NotImplementedError("Labels must be specified manually or via the defaults provided in the CLI")
        else:
            raise RuntimeError("Labels or a labels file must be specified for AddClassificationStage")

    def _add_labels(self, x: MultiResponseProbsMessage, idx2label: typing.Mapping[int, str]):

        if (x.probs.shape[1] != len(idx2label)):
            raise RuntimeError("Label count does not match output of model. Label count: {}, Model output: {}".format(
                len(idx2label), x.probs.shape[1]))

        probs_np = (x.probs > self._threshold).astype(cp.bool).get()

        for i, label in idx2label.items():
            x.set_meta(label, probs_np[:, i].tolist())

        # Return list of strs to write out
        return x

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        # First, determine the labels
        labels = self._determine_labels()

        assert len(labels) > 0, "Labels must be non-zero array"

        idx2label = {i: self._prefix + l for i, l in enumerate(labels)}

        stream = input_stream[0]

        # Convert the messages to rows of strings
        stream = stream.async_map(self._add_labels, executor=self._pipeline.thread_pool, idx2label=idx2label)

        # Return input unchanged
        return stream, MultiResponseProbsMessage


class FilterDetectionsStage(SinglePortStage):
    """
    This Stage class is used to filter results based on a given criteria.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    threshold : float
        Threshold to classify, default is 0.5

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
        typing.Tuple[MultiResponseProbsMessage, ]
            Accepted input types

        """
        return (MultiResponseProbsMessage, )

    def filter(self, x: MultiResponseProbsMessage) -> typing.List[MultiResponseProbsMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : morpheus.messages.MultiResponseProbsMessage
            MultiResponseProbsMessage

        Returns
        -------
        typing.List[MultiResponseProbsMessage]
            list of filtered messages

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

            output_list.append(
                MultiResponseProbsMessage(x.meta,
                                          mess_offset=mess_offset,
                                          mess_count=mess_count,
                                          memory=x.memory,
                                          offset=pair[0],
                                          count=mess_count))

        return output_list

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Reduce messages to only have detections
        stream = stream.async_map(self.filter, executor=self._pipeline.thread_pool)

        # Convert list back to single MultiResponseProbsMessage
        stream = stream.flatten()

        # Filter out empty message groups
        stream = stream.filter(lambda x: x.count > 0)

        return stream, MultiResponseProbsMessage


class ZipStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "zip"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _build(self, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        # Check for compatible types
        stream_types = [s_type for s, s_type in in_ports_streams[1:]]

        # Find greatest_ancestor
        out_type = greatest_ancestor(*stream_types)

        if (out_type is None):
            out_type = unpack_union(*stream_types)

        # Build off first stream
        first_pair = in_ports_streams[0]

        first_stream = first_pair[0]

        stream = first_stream.zip([s for s, _ in in_ports_streams[1:]])

        return [(stream, out_type)]


class MergeStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "merge"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _build(self, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        # Check for compatible types
        stream_types = [s_type for s, s_type in in_ports_streams]

        # Find greatest_ancestor
        out_type = greatest_ancestor(*stream_types)

        if (out_type is None):
            out_type = unpack_tuple(*stream_types)

        stream = streamz.union(*[s for s, _ in in_ports_streams])

        return [(stream, out_type)]


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

    def _build(self, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        # Since we are a SiSo stage, there will only be 1 input
        input_stream = in_ports_streams[0]

        stream = input_stream[0]

        # Filter out empty message groups
        switch_stream = stream.switch(self._predicate)

        out_pairs = []

        for _ in range(self._num_outputs):
            out_pairs.append((Stream(upstream=switch_stream), input_stream[1]))

        return out_pairs
