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

import asyncio
import collections
import inspect
import logging
import os
import signal
import time
import typing
from abc import ABC
from abc import abstractmethod

import neo
import networkx
import typing_utils
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import MultiMessage
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import pretty_print_type_name

config = Config()

logger = logging.getLogger(__name__)


def get_time_ms():
    return round(time.time() * 1000)


T = typing.TypeVar('T')

StreamPair = typing.Tuple[neo.Node, typing.Type]


class Sender():
    """
    The `Sender` object represents a port on a `StreamWrapper` object that sends messages to a `Receiver`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StreamWrapper`
            Parent `StreamWrapper` object.
        port_number : int
            Sender port number.
    """

    def __init__(self, parent: "StreamWrapper", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._output_receivers: typing.List[Receiver] = []

        self._out_stream_pair: StreamPair = (None, None)

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        # Sender is complete when the type or stream has been set
        return self._out_stream_pair != (None, None)

    @property
    def out_pair(self):
        return self._out_stream_pair

    @property
    def out_stream(self):
        return self._out_stream_pair[0]

    @property
    def out_type(self):
        return self._out_stream_pair[1]


class Receiver():
    """
    The `Receiver` object represents a downstream port on a `StreamWrapper` object that gets messages from a `Sender`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StreamWrapper`
            Parent `StreamWrapper` object.
        port_number : int
            Receiver port number.
    """

    def __init__(self, parent: "StreamWrapper", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._is_linked = False

        self._input_type = None
        self._input_stream = None

        self._input_senders: typing.List[Sender] = []

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        """
        A receiver is complete if all input senders are also complete.
        """
        return all([x.is_complete for x in self._input_senders])

    @property
    def is_partial(self):
        """
        A receiver is partially complete if any input sender is complete. Receivers are usually partially complete if
        there is a circular pipeline.
        """
        # Its partially complete if any input sender is complete
        return any([x.is_complete for x in self._input_senders])

    @property
    def in_pair(self):
        return (self.in_stream, self.in_pair)

    @property
    def in_stream(self):
        return self._input_stream

    @property
    def in_type(self):
        return self._input_type

    def get_input_pair(self) -> StreamPair:

        assert self.is_partial, "Must be partially complete to get the input pair!"

        # Build the input from the senders
        if (self._input_stream is None and self._input_type is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input stream/type is determined from the sole Sender
                sender = self._input_senders[0]

                self._input_stream = sender.out_stream
                self._input_type = sender.out_type
                self._is_linked = True
            else:
                # We have multiple senders. Create a dummy stream to connect all senders
                if (self.is_complete):
                    # Connect all streams now
                    # self._input_stream = streamz.Stream(upstreams=[x.out_stream for x in self._input_senders],
                    #                                     asynchronous=True,
                    #                                     loop=IOLoop.current())
                    raise NotImplementedError("Still using streamz")
                    self._is_linked = True
                else:
                    # Create a dummy stream that needs to be linked later
                    # self._input_stream = streamz.Stream(asynchronous=True, loop=IOLoop.current())
                    raise NotImplementedError("Still using streamz")

                # Now determine the output type from what we have
                great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

                if (great_ancestor is None):
                    # TODO: Add stage, port, and type info to message
                    raise RuntimeError(("Cannot determine single type for senders of input port. "
                                        "Use a merge stage to handle different types of inputs."))

                self._input_type = great_ancestor

        return (self._input_stream, self._input_type)

    def link(self):
        """
        The linking phase determines the final type of the `Receiver` and connects all underlying stages.

        Raises:
            RuntimeError: Throws a `RuntimeError` if the predicted input port type determined during the build phase is
            different than the current port type.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_linked):
            return

        # Check that the types still work
        great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

        if (not typing_utils.issubtype(great_ancestor, self._input_type)):
            # TODO: Add stage, port, and type info to message
            raise RuntimeError(
                "Invalid linking phase. Input port type does not match predicted type determined during build phase")

        for out_stream in [x.out_stream for x in self._input_senders]:
            out_stream.connect(self._input_stream)

        self._is_linked = True


def _save_init_vals(func: _DecoratorType) -> _DecoratorType:

    # Save the signature only once
    sig = inspect.signature(func, follow_wrapped=True)

    def inner(self: "StreamWrapper", c: Config, *args, **kwargs):

        # Actually call init first. This way any super classes strings will be overridden
        func(self, c, *args, **kwargs)

        # Determine all set values
        bound = sig.bind(self, c, *args, **kwargs)
        bound.apply_defaults()

        init_pairs = []

        for key, val in bound.arguments.items():

            # We really dont care about these
            if (key == "self" or key == "c"):
                continue

            init_pairs.append(f"{key}={val}")

        # Save values on self
        self._init_str = ", ".join(init_pairs)

        return

    return typing.cast(_DecoratorType, inner)


class StreamWrapper(ABC, collections.abc.Hashable):
    """
    This abstract class serves as the morpheus.pipeline's base class. This class wraps a `neo.Node`
    object and aids in hooking stages up together.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    __ID_COUNTER = AtomicInteger(0)

    def __init__(self, c: Config):
        self._id = StreamWrapper.__ID_COUNTER.get_and_inc()
        self._pipeline: Pipeline = None
        self._init_str: str = ""  # Stores the initialization parameters used for creation. Needed for __repr__

        # Indicates whether or not this wrapper has been built. Can only be built once
        self._is_built = False

        # Input/Output ports used for connecting stages
        self._input_ports: typing.List[Receiver] = []
        self._output_ports: typing.List[Sender] = []

    def __init_subclass__(cls) -> None:

        # Wrap __init__ to save the arg values
        cls.__init__ = _save_init_vals(cls.__init__)

        return super().__init_subclass__()

    def __hash__(self) -> int:
        return self._id

    def __str__(self):
        text = f"<{self.unique_name}; {self.__class__.__name__}({self._init_str})>"

        return text

    __repr__ = __str__

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the stage. Used in logging. Each derived class should override this property with a unique
        name.

        Returns
        -------
        str
            Name of a stage.

        """
        pass

    @property
    def unique_name(self) -> str:
        """
        Unique name of stag. Generated by appending stage id to stage name.

        Returns
        -------
        str
            Unique name of stage.
        """
        return f"{self.name}-{self._id}"

    @property
    def is_built(self) -> bool:
        """
        Indicates if this stage has been built.

        Returns
        -------
        bool
            True if stage is built, False otherwise.
        """
        return self._is_built

    @property
    def input_ports(self) -> typing.List[Receiver]:
        """Input ports to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Receiver`]
            Input ports to this stage.
        """
        return self._input_ports

    @property
    def output_ports(self) -> typing.List[Sender]:
        """
        Output ports from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Sender`]
            Output ports from this stage.
        """
        return self._output_ports

    @property
    def has_multi_input_ports(self) -> bool:
        """
        Indicates if this stage has multiple input ports.

        Returns
        -------
        bool
            True if stage has multiple input ports, False otherwise.
        """
        return len(self._input_ports) > 1

    @property
    def has_multi_output_ports(self) -> bool:
        """
        Indicates if this stage has multiple output ports.

        Returns
        -------
        bool
            True if stage has multiple output ports, False otherwise.
        """
        return len(self._output_ports) > 1

    def get_all_inputs(self) -> typing.List[Sender]:
        """
        Get all input senders to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Sender`]
            All input senders.
        """

        senders = []

        for in_port in self._input_ports:
            senders.extend(in_port._input_senders)

        return senders

    def get_all_input_stages(self) -> typing.List["StreamWrapper"]:
        """
        Get all input stages to this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.StreamWrapper`]
            All input stages.
        """
        return [x.parent for x in self.get_all_inputs()]

    def get_all_outputs(self) -> typing.List[Receiver]:
        """
        Get all output receivers from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.Receiver`]
            All output receivers.
        """
        receivers = []

        for out_port in self._output_ports:
            receivers.extend(out_port._output_receivers)

        return receivers

    def get_all_output_stages(self) -> typing.List["StreamWrapper"]:
        """
        Get all output stages from this stage.

        Returns
        -------
        typing.List[`morpheus.pipeline.pipeline.StreamWrapper`]
            All output stages.
        """
        return [x.parent for x in self.get_all_outputs()]

    def supports_cpp_node(self):
        """
        Specifies whether this Stage is even capable of creating C++ nodes. During the build phase, this value will be
        combined with Config.get().use_cpp to determine whether or not a C++ node is created. This is an instance method
        to allow runtime decisions and derived classes to override base implementations.
        """
        # By default, return False unless otherwise specified
        return False

    def _build_cpp_node(self):
        """
        Specifies whether or not to build a C++ node. Only should be called during the build phase.
        """
        return CppConfig.get_should_use_cpp() and self.supports_cpp_node()

    def can_build(self, check_ports=False) -> bool:
        """
        Determines if all inputs have been built allowing this node to be built.

        Parameters
        ----------
        check_ports : bool, optional
            Check if we can build based on the input ports, by default False.

        Returns
        -------
        bool
            True if we can build, False otherwise.
        """

        # Can only build once
        if (self.is_built):
            return False

        if (not check_ports):
            # We can build if all input stages have been built. Easy and quick check. Works for non-circular pipelines
            for in_stage in self.get_all_input_stages():
                if (not in_stage.is_built):
                    return False

            return True
        else:
            # Check if we can build based on the input ports. We can build
            for r in self.input_ports:
                if (not r.is_partial):
                    return False

            return True

    def build(self, seg: neo.Segment, do_propagate=True):
        """Build this stage.

        Parameters
        ----------
        seg : `neo.Segment`
            Neo segment for this stage.
        do_propagate : bool, optional
            Whether to propagate to build output stages, by default True.

        """
        assert not self.is_built, "Can only build stages once!"
        assert self._pipeline is not None, "Must be attached to a pipeline before building!"

        # Pre-Build returns the input pairs for each port
        in_ports_pairs = self._pre_build()

        out_ports_pair = self._build(seg, in_ports_pairs)

        # Allow stages to do any post build steps (i.e., for sinks, or timing functions)
        out_ports_pair = self._post_build(seg, out_ports_pair)

        assert len(out_ports_pair) == len(self.output_ports), \
            "Build must return same number of output pairs as output ports"

        # Assign the output ports
        for port_idx, out_pair in enumerate(out_ports_pair):
            self.output_ports[port_idx]._out_stream_pair = out_pair

        self._is_built = True

        if (not do_propagate):
            return

        # Now build for any dependents
        for dep in self.get_all_output_stages():
            if (not dep.can_build()):
                continue

            dep.build(seg, do_propagate=do_propagate)

    def _pre_build(self) -> typing.List[StreamPair]:
        in_pairs: typing.List[StreamPair] = [x.get_input_pair() for x in self.input_ports]

        return in_pairs

    @abstractmethod
    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        """
        This function is responsible for constructing this stage's internal `neo.Node` object. The input
        of this function contains the returned value from the upstream stage.

        The input values are the `neo.Segment` for this stage and a `StreamPair` tuple which contain the input
        `neo.Node` object and the message data type.

        :meta public:

        Parameters
        ----------
        seg : `neo.Segment`
            `neo.Segment` object for the pipeline. This should be used to construct/attach the internal `neo.Node`.
        in_ports_streams : `morpheus.pipeline.pipeline.StreamPair`
            List of tuples containing the input `neo.Node` object and the message data type.

        Returns
        -------
        `typing.List[morpheus.pipeline.pipeline.StreamPair]`
            List of tuples containing the output `neo.Node` object from this stage and the message data type.

        """
        pass

    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:
        return out_ports_pair

    def start(self):

        assert self.is_built, "Must build before starting!"

        self._start()

    def _start(self):
        pass

    def stop(self):
        """
        Stages can implement this to perform cleanup steps when pipeline is stopped.
        """
        pass

    async def join(self):
        pass

    def _create_ports(self, input_count: int, output_count: int):
        assert len(self._input_ports) == 0 and len(self._output_ports) == 0, "Can only create ports once!"

        self._input_ports = [Receiver(parent=self, port_number=i) for i in range(input_count)]
        self._output_ports = [Sender(parent=self, port_number=i) for i in range(output_count)]


class SourceStage(StreamWrapper):
    """
    The SourceStage is mandatory for the Morpheus pipeline to run. This stage represents the start of the pipeline. All
    `SourceStage` object take no input but generate output.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._start_callbacks: typing.List[typing.Callable] = []
        self._stop_callbacks: typing.List[typing.Callable] = []

        self._source_stream: neo.Node = None

    @property
    def input_count(self) -> int:
        """
        Return None for no max intput count.

        Returns
        -------
        int
            Input count.

        """
        return None

    @abstractmethod
    def _build_source(self, seg: neo.Segment) -> StreamPair:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`.

        :meta public:

        Returns
        -------

        `morpheus.pipeline.pipeline.StreamPair`:
            A tuple containing the output `neo.Node` object from this stage and the message data type.
        """

        pass

    @typing.final
    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 0, "Sources shouldnt have input ports"

        source_pair = self._build_source(seg)

        curr_source = source_pair[0]

        self._source_stream = curr_source

        # Now setup the output ports
        self._output_ports[0]._out_stream_pair = source_pair

        return [source_pair]

    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        return out_ports_pair

    def _start(self):
        self._source_stream.start()

    def stop(self):
        self._source_stream.stop()

    async def join(self):
        self._source_stream.join()


class SingleOutputSource(SourceStage):
    """
    Subclass of SourceStage for building source stages that generate output for single port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(0, 1)

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:
        return out_pair

    @typing.final
    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        ret_val = self._post_build_single(seg, out_ports_pair[0])

        logger.info("Added source: {}\n  └─> {}".format(str(self), pretty_print_type_name(ret_val[1])))

        return [ret_val]


class Stage(StreamWrapper):
    """
    This class serves as the base for all pipeline stage implementations that are not source objects.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        return out_ports_pair

    def _start(self):
        pass

    def on_start(self):
        """
        This function can be overridden to add usecase-specific implementation at the start of any stage in
        the pipeline.
        """
        pass

    def _on_complete(self, stream):

        logger.info("Stage Complete: {}".format(self.name))


class SinglePortStage(Stage):
    """
    Class used for building stages with single input port and single output port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 1)

    @abstractmethod
    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned. Derived classes should override this method. An
        error will be generated if the input types to the stage do not match one of the available types
        returned from this method.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        pass

    def _pre_build(self) -> typing.List[StreamPair]:
        in_ports_pairs = super()._pre_build()

        # Check the types of all inputs
        for x in in_ports_pairs:
            if (not typing_utils.issubtype(x[1], typing.Union[self.accepted_types()])):
                raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
                    self.name, x[1], self.accepted_types()))

        return in_ports_pairs

    @abstractmethod
    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        pass

    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 1 and len(self.output_ports) == 1, \
            "SinglePortStage must have 1 input port and 1 output port"

        assert len(in_ports_streams) == 1, "Should only have 1 port on input"

        return [self._build_single(seg, in_ports_streams[0])]

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:
        return out_pair

    @typing.final
    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        ret_val = self._post_build_single(seg, out_ports_pair[0])

        logger.info("Added stage: {}\n  └─ {} -> {}".format(str(self),
                                                            pretty_print_type_name(self.input_ports[0].in_type),
                                                            pretty_print_type_name(ret_val[1])))

        return [ret_val]


class MultiMessageStage(SinglePortStage):
    """
    Subclass of SinglePortStage with option to log timestamps in MessageMeta dataframe.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):

        # Derived classes should set this to true to log timestamps in debug builds
        self._should_log_timestamps = False

        super().__init__(c)

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        # Check if we are debug and should log timestamps
        if (config.debug and self._should_log_timestamps):
            # Cache the name property. Removes dependency on self in callback
            cached_name = self.name

            logger.info("Adding timestamp info for stage: '%s'", cached_name)

            def post_timestamps(x: MultiMessage):

                curr_time = get_time_ms()

                x.set_meta("_ts_" + cached_name, curr_time)

            # Only have one port
            post_ts = seg.make_node(self.unique_name + "-ts", post_timestamps)
            seg.make_edge(out_pair[0], post_ts)

            # Keep the type unchanged
            out_pair = (post_ts, out_pair[1])

        return super()._post_build_single(seg, out_pair)


class Pipeline():
    """
    Class for building your pipeline. A pipeline for your use case can be constructed by first adding a
    `Source` via `set_source` then any number of downstream `Stage` classes via `add_stage`. The order stages
    are added with `add_stage` determines the order in which stage executions are carried out. You can use
    stages included within Morpheus or your own custom-built stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0

        self._sources: typing.Set[SourceStage] = set()
        self._stages: typing.Set[Stage] = set()

        self._exec_options = neo.Options()
        self._exec_options.topology.user_cpuset = "0-{}".format(c.num_threads - 1)

        # Set the default channel size
        neo.Config.default_channel_size = c.edge_buffer_size

        self.batch_size = c.pipeline_batch_size

        self._graph = networkx.DiGraph()

        self._is_built = False
        self._is_started = False

        self._neo_executor: neo.Executor = None
        self._neo_pipeline: neo.Pipeline = None

    @property
    def is_built(self) -> bool:
        return self._is_built

    def _add_id_col(self, x: cudf.DataFrame):

        # Data in stream is cudf Dataframes at this point. We need an ID column before continuing
        x.insert(0, 'ID', range(self._id_counter, self._id_counter + len(x)))
        self._id_counter += len(x)

        return x

    def add_node(self, node: StreamWrapper):

        assert node._pipeline is None or node._pipeline is self, "A stage can only be added to one pipeline at a time"

        # Add to list of stages if its a stage, not a source
        if (isinstance(node, Stage)):
            self._stages.add(node)
        elif (isinstance(node, SourceStage)):
            self._sources.add(node)
        else:
            raise NotImplementedError("add_node() failed. Unknown node type: {}".format(type(node)))

        node._pipeline = self

        self._graph.add_node(node)

    def add_edge(self, start: typing.Union[StreamWrapper, Sender], end: typing.Union[Stage, Receiver]):

        if (isinstance(start, StreamWrapper)):
            start_port = start.output_ports[0]
        elif (isinstance(start, Sender)):
            start_port = start

        if (isinstance(end, Stage)):
            end_port = end.input_ports[0]
        elif (isinstance(end, Receiver)):
            end_port = end

        # Ensure both are added to this pipeline
        self.add_node(start_port.parent)
        self.add_node(end_port.parent)

        start_port._output_receivers.append(end_port)
        end_port._input_senders.append(start_port)

        self._graph.add_edge(start_port.parent,
                             end_port.parent,
                             start_port_idx=start_port.port_number,
                             end_port_idx=end_port.port_number)

    def build(self):
        """
        This function sequentially activates all of the Morpheus pipeline stages passed by the users to execute a
        pipeline. For the `Source` and all added `Stage` objects, `StreamWrapper.build` will be called sequentially to
        construct the pipeline.

        Once the pipeline has been constructed, this will start the pipeline by calling `Source.start` on the source
        object.
        """
        assert not self._is_built, "Pipeline can only be built once!"
        assert len(self._sources) > 0, "Pipeline must have a source stage"

        logger.info("====Registering Pipeline====")

        self._neo_executor = neo.Executor(self._exec_options)

        self._neo_pipeline = neo.Pipeline()

        def inner_build(seg: neo.Segment):
            logger.info("====Building Pipeline====")

            # Get the list of stages and source
            source_and_stages: typing.List[StreamWrapper] = list(self._sources) + list(self._stages)

            # Now loop over stages
            for s in source_and_stages:

                if (s.can_build()):
                    s.build(seg)

            if (not all([x.is_built for x in source_and_stages])):
                # raise NotImplementedError("Circular pipelines are not yet supported!")
                logger.warning("Circular pipeline detected! Building with reduced constraints")

                for s in source_and_stages:

                    if (s.can_build(check_ports=True)):
                        s.build()

            if (not all([x.is_built for x in source_and_stages])):
                raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            for s in source_and_stages:

                for p in s.input_ports:
                    p.link()

            logger.info("====Building Pipeline Complete!====")

            # Finally call _on_start
            self._on_start()

        self._neo_pipeline.make_segment("main", inner_build)

        self._neo_executor.register_pipeline(self._neo_pipeline)

        self._is_built = True

        logger.info("====Registering Pipeline Complete!====")

    def start(self):
        assert self._is_built, "Pipeline must be built before starting"

        logger.info("====Starting Pipeline====")

        self._neo_executor.start()

        logger.info("====Pipeline Started====")

    def stop(self):

        logger.info("====Stopping Pipeline====")
        for s in list(self._sources) + list(self._stages):
            s.stop()

        self._neo_executor.stop()

        logger.info("====Pipeline Stopped====")

    async def join(self):

        await self._neo_executor.join_async()

        # First wait for all sources to stop. This only occurs after all messages have been processed fully
        for s in list(self._sources):
            await s.join()

        # Now that there is no more data, call stop on all stages to ensure shutdown (i.e., for stages that have their
        # own worker loop thread)
        for s in list(self._stages):
            s.stop()

        # Now call join on all stages
        for s in list(self._stages):
            await s.join()

    def build_and_start(self):

        if (not self.is_built):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                return

        self.start()

    def _on_start(self):

        # Only execute this once
        if (self._is_started):
            return

        # Stop from running this twice
        self._is_started = True

        logger.debug("Starting! Time: {}".format(time.time()))

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            s.on_start()

    def visualize(self, filename: str = None, **graph_kwargs):

        # Mimic the streamz visualization
        # 1. Create graph (already done since we use networkx under the hood)
        # 2. Readable graph
        # 3. To graphviz
        # 4. Draw
        import graphviz

        # Default graph attributes
        graph_attr = {
            "nodesep": "1",
            "ranksep": "1",
            "pad": "0.5",
        }

        # Allow user to overwrite defaults
        graph_attr.update(graph_kwargs)

        gv_graph = graphviz.Digraph(graph_attr=graph_attr)

        # Need a little different functionality for left/right vs vertical
        is_lr = graph_kwargs.get("rankdir", None) == "LR"

        start_def_port = ":e" if is_lr else ":s"
        end_def_port = ":w" if is_lr else ":n"

        def has_ports(n: StreamWrapper, is_input):
            if (is_input):
                return len(n.input_ports) > 0
            else:
                return len(n.output_ports) > 0

        # Now build up the nodes
        for n, attrs in typing.cast(typing.Mapping[StreamWrapper, dict], self._graph.nodes).items():
            node_attrs = attrs.copy()

            label = ""

            show_in_ports = has_ports(n, is_input=True)
            show_out_ports = has_ports(n, is_input=False)

            # Build the ports for the node. Only show ports if there are any (Would like to have this not show for one
            # port, but the lines get all messed up)
            if (show_in_ports):
                in_port_label = " {{ {} }} | ".format(" | ".join(
                    [f"<u{x.port_number}> {x.port_number}" for x in n.input_ports]))
                label += in_port_label

            label += n.unique_name

            if (show_out_ports):
                out_port_label = " | {{ {} }}".format(" | ".join(
                    [f"<d{x.port_number}> {x.port_number}" for x in n.output_ports]))
                label += out_port_label

            if (show_in_ports or show_out_ports):
                label = f"{{ {label} }}"

            node_attrs.update({
                "label": label,
                "shape": "record",
                "fillcolor": "white",
            })
            # TODO: Eventually allow nodes to have different attributes based on type
            # node_attrs.update(n.get_graphviz_attrs())
            gv_graph.node(n.unique_name, **node_attrs)

        # Build up edges
        for e, attrs in typing.cast(typing.Mapping[typing.Tuple[StreamWrapper, StreamWrapper], dict], self._graph.edges()).items():  # noqa: E501

            edge_attrs = {}

            start_name = e[0].unique_name

            # Append the port if necessary
            if (has_ports(e[0], is_input=False)):
                start_name += f":d{attrs['start_port_idx']}"
            else:
                start_name += start_def_port

            end_name = e[1].unique_name

            if (has_ports(e[1], is_input=True)):
                end_name += f":u{attrs['end_port_idx']}"
            else:
                end_name += end_def_port

            # Now we only want to show the type label in some scenarios:
            # 1. If there is only one edge between two nodes, draw type in middle "label"
            # 2. For port with an edge, only draw that port's type once (using index 0 of the senders/receivers)
            start_port_idx = int(attrs['start_port_idx'])
            end_port_idx = int(attrs['end_port_idx'])

            out_port = e[0].output_ports[start_port_idx]
            in_port = e[1].input_ports[end_port_idx]

            # Check for situation #1
            if (len(in_port._input_senders) == 1 and len(out_port._output_receivers) == 1
                    and (in_port.in_type == out_port.out_type)):

                edge_attrs["label"] = pretty_print_type_name(in_port.in_type)
            else:
                rec_idx = out_port._output_receivers.index(in_port)
                sen_idx = in_port._input_senders.index(out_port)

                # Add type labels if available
                if (rec_idx == 0 and out_port.out_type is not None):
                    edge_attrs["taillabel"] = pretty_print_type_name(out_port.out_type)

                if (sen_idx == 0 and in_port.in_type is not None):
                    edge_attrs["headlabel"] = pretty_print_type_name(in_port.in_type)

            gv_graph.edge(start_name, end_name, **edge_attrs)

        file_format = os.path.splitext(filename)[-1].replace(".", "")

        viz_binary = gv_graph.pipe(format=file_format)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            f.write(viz_binary)

    async def _do_run(self):
        """
        This function sets up the current asyncio loop, builds the pipeline, and awaits on it to complete.
        """
        loop = asyncio.get_running_loop()

        def error_handler(_, context: dict):

            msg = "Unhandled exception in async loop! Exception: \n{}".format(context["message"])
            exception = context.get("exception", Exception())

            logger.critical(msg, exc_info=exception)

        loop.set_exception_handler(error_handler)

        exit_count = 0

        # Handles Ctrl+C for graceful shutdown
        def term_signal():

            nonlocal exit_count
            exit_count = exit_count + 1

            if (exit_count == 1):
                tqdm.write("Stopping pipeline. Please wait... Press Ctrl+C again to kill.")
                self.stop()
            else:
                tqdm.write("Killing")
                exit(1)

        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, term_signal)

        try:
            self.build_and_start()

            # Wait for completion
            await self.join()

        except KeyboardInterrupt:
            tqdm.write("Stopping pipeline. Please wait...")

            # Stop the pipeline
            self.stop()

            # Wait again for nice completion
            await self.join()

        finally:
            # Shutdown the async generator sources and exit
            logger.info("====Pipeline Complete====")

    def run(self):
        """
        This function makes use of asyncio features to keep the pipeline running indefinitely.
        """

        # Use asyncio.run() to launch the pipeline. This creates and destroys an event loop so re-running a pipeline in
        # the same process wont fail
        asyncio.run(self._do_run())


class LinearPipeline(Pipeline):
    """
    This class is used to build linear pipelines where we have a single output source stage followed by stages that are
    executed sequentially in the order they were added.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._linear_stages: typing.List[StreamWrapper] = []

    def set_source(self, source: SourceStage):
        """
        Set a pipeline's source stage to consume messages before it begins executing stages. This must be
        called once before `build_and_start`.

        Parameters
        ----------
        source : `SourceStage`
            The source stage wraps the implementation in a stream that allows it to read from Kafka or a file.

        """

        if (len(self._sources) > 0 and source not in self._sources):
            logger.warning(
                "LinearPipeline already has a source. Setting a new source will clear out all existing stages")

            self._sources.clear()

        # Store the source in sources
        self._sources.add(source)

        if (len(self._linear_stages) > 0):
            logger.warning("Clearing %d stages from pipeline", len(self._linear_stages))
            self._linear_stages.clear()

        # Need to store the source in the pipeline
        super().add_node(source)

        # Store this as the first one in the linear stages. Must be index 0
        self._linear_stages.append(source)

    def add_stage(self, stage: SinglePortStage):
        """
        Add stages to the pipeline. All `Stage` classes added with this method will be executed sequentially
        inthe order they were added.

        Parameters
        ----------
        stage : `Stage`
            The stage object to add. It cannot be already added to another `Pipeline` object.

        """

        assert len(self._linear_stages) > 0, "A source must be set on a LinearPipeline before adding any stages"
        assert isinstance(stage, SinglePortStage), "Only `SinglePortStage` stages are accepted in `add_stage()`"

        # Make an edge between the last node and this one
        self.add_edge(self._linear_stages[-1], stage)

        self._linear_stages.append(stage)
