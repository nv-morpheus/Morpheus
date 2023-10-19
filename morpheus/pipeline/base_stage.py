# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import functools
import inspect
import logging
import typing
from abc import ABC
from abc import abstractmethod

import mrc

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType

logger = logging.getLogger(__name__)


def _save_init_vals(func: _DecoratorType) -> _DecoratorType:

    # Save the signature only once
    sig = inspect.signature(func, follow_wrapped=True)

    @functools.wraps(func)
    def inner(self: "StageBase", *args, **kwargs):

        # Actually call init first. This way any super classes strings will be overridden
        func(self, *args, **kwargs)

        # Determine all set values
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        init_pairs = []

        for key, val in bound.arguments.items():

            # We really dont care about these
            if (key == "self" or sig.parameters[key].annotation == Config):
                continue

            init_pairs.append(f"{key}={val}")

        # Save values on self
        self._init_str = ", ".join(init_pairs)

    return typing.cast(_DecoratorType, inner)


class StageBase(ABC, collections.abc.Hashable):
    """
    This abstract class serves as the morpheus pipeline's base class. This class wraps a `mrc.SegmentObject`
    object and aids in hooking stages up together.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    __ID_COUNTER = AtomicInteger(0)

    def __init__(self, config: Config):
        # Save the config
        self._config = config

        self._id = StageBase.__ID_COUNTER.get_and_inc()
        self._pipeline: _pipeline.Pipeline = None
        self._init_str: str = ""  # Stores the initialization parameters used for creation. Needed for __repr__

        # Indicates whether or not this wrapper has been built. Can only be built once
        self._is_pre_built = False
        self._is_built = False

        # Input/Output ports used for connecting stages
        self._input_ports: list[_pipeline.Receiver] = []
        self._output_ports: list[_pipeline.Sender] = []

        # Mapping of {`column_name`: `TyepId`}
        self._needed_columns = collections.OrderedDict()

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
        Unique name of stage. Generated by appending stage id to stage name.

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
    def is_pre_built(self) -> bool:
        """
        Indicates if this stage has been built.

        Returns
        -------
        bool
            True if stage is built, False otherwise.
        """
        return self._is_pre_built

    @property
    def input_ports(self) -> list[_pipeline.Receiver]:
        """Input ports to this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.Receiver`]
            Input ports to this stage.
        """
        return self._input_ports

    @property
    def output_ports(self) -> list[_pipeline.Sender]:
        """
        Output ports from this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.Sender`]
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

    def get_all_inputs(self) -> list[_pipeline.Sender]:
        """
        Get all input senders to this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.Sender`]
            All input senders.
        """

        senders = []

        for in_port in self._input_ports:
            senders.extend(in_port._input_senders)

        return senders

    def get_all_input_stages(self) -> list["StageBase"]:
        """
        Get all input stages to this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.BaseStage`]
            All input stages.
        """
        return [x.parent for x in self.get_all_inputs()]

    def get_all_outputs(self) -> list[_pipeline.Receiver]:
        """
        Get all output receivers from this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.Receiver`]
            All output receivers.
        """
        receivers = []

        for out_port in self._output_ports:
            receivers.extend(out_port._output_receivers)

        return receivers

    def get_all_output_stages(self) -> list["StageBase"]:
        """
        Get all output stages from this stage.

        Returns
        -------
        list[`morpheus.pipeline.pipeline.BaseStage`]
            All output stages.
        """
        return [x.parent for x in self.get_all_outputs()]

    @abstractmethod
    def supports_cpp_node(self):
        """
        Specifies whether this Stage is capable of creating C++ nodes. During the build phase, this value will be
        combined with `CppConfig.get_should_use_cpp()` to determine whether or not a C++ node is created. This is an
        instance method to allow runtime decisions and derived classes to override base implementations.
        """
        # By default, return False unless otherwise specified
        # return False
        pass

    def _build_cpp_node(self):
        """
        Specifies whether or not to build a C++ node. Only should be called during the build phase.
        """
        return CppConfig.get_should_use_cpp() and self.supports_cpp_node()

    def can_pre_build(self, check_ports=False) -> bool:
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
        if (self.is_pre_built):
            return False

        if (not check_ports):
            # We can prebuild if all input stages have been prebuilt.
            # Easy and quick check. Works for non-circular pipelines
            return all(stage.is_pre_built for stage in self.get_all_input_stages())

        # Check if we can prebuild based on the input ports.
        return all(receiver.is_partial for receiver in self.input_ports)

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
            return all(stage.is_built for stage in self.get_all_input_stages())

        # Check if we can build based on the input ports. We can build
        return all(receiver.is_partial for receiver in self.input_ports)

    def _pre_build(self, do_propagate: bool = True):
        assert not self.is_built, "build called prior to _pre_build"
        assert not self.is_pre_built, "Can only pre-build stages once!"
        schema = _pipeline.StageSchema(self)
        self._pre_compute_schema(schema)
        self.compute_schema(schema)

        assert len(schema.output_schemas) == len(self.output_ports), \
            (f"Prebuild expected `schema.output_schemas` to be of length {len(self.output_ports)} "
             f"(one for each output port), but got {len(schema.output_schemas)}.")

        schema._complete()

        for (port_idx, port_schema) in enumerate(schema.output_schemas):
            self.output_ports[port_idx].output_schema = port_schema

        self._is_pre_built = True

        if (not do_propagate):
            return

        # Now pre-build for any dependents
        for dep in self.get_all_output_stages():
            if (not dep.can_pre_build()):
                continue

            dep._pre_build(do_propagate=do_propagate)

    def build(self, builder: mrc.Builder, do_propagate: bool = True):
        """Build this stage.

        Parameters
        ----------
        builder : `mrc.Builder`
            MRC segment for this stage.
        do_propagate : bool, optional
            Whether to propagate to build output stages, by default True.

        """
        assert self._is_pre_built, "Must be pre-built before building!"
        assert not self.is_built, "Can only build stages once!"
        assert self._pipeline is not None, "Must be attached to a pipeline before building!"

        in_ports_nodes = [x.get_input_node(builder=builder) for x in self.input_ports]

        out_ports_nodes = self._build(builder=builder, input_nodes=in_ports_nodes)

        # Allow stages to do any post build steps (i.e., for sinks, or timing functions)
        out_ports_nodes = self._post_build(builder=builder, out_ports_nodes=out_ports_nodes)

        assert len(out_ports_nodes) == len(self.output_ports), \
            "Build must return same number of output pairs as output ports"

        # Assign the output ports
        for port_idx, out_node in enumerate(out_ports_nodes):
            self.output_ports[port_idx]._output_node = out_node

        self._is_built = True

        if (not do_propagate):
            return

        # Now build for any dependents
        for dep in self.get_all_output_stages():
            if (not dep.can_build()):
                continue

            dep.build(builder, do_propagate=do_propagate)

    @abstractmethod
    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        """
        This function is responsible for constructing this stage's internal `mrc.SegmentObject` object. The input
        of this function contains the returned value from the upstream stage.

        The input values are the `mrc.Builder` for this stage and a list of parent nodes.

        :meta public:

        Parameters
        ----------
        builder : `mrc.Builder`
            `mrc.Builder` object for the pipeline. This should be used to construct/attach the internal
            `mrc.SegmentObject`.
        input_nodes : `list[mrc.SegmentObject]`
            List containing the input `mrc.SegmentObject` objects.

        Returns
        -------
        `list[mrc.SegmentObject]`
            List of tuples containing the output `mrc.SegmentObject` object from this stage.

        """
        pass

    def _post_build(
        self,
        builder: mrc.Builder,  # pylint: disable=unused-argument
        out_ports_nodes: list[mrc.SegmentObject],
    ) -> list[mrc.SegmentObject]:
        return out_ports_nodes

    def _start(self):
        pass

    def stop(self):
        """
        Stages can implement this to perform cleanup steps when pipeline is stopped.
        """
        pass

    async def join(self):
        """
        Awaitable method that stages can implement this to perform cleanup steps when pipeline is stopped.
        Typically this is called after `stop` during a graceful shutdown, but may not be called if the pipeline is
        terminated.
        """
        pass

    def _create_ports(self, input_count: int, output_count: int):
        assert len(self._input_ports) == 0 and len(self._output_ports) == 0, "Can only create ports once!"

        self._input_ports = [_pipeline.Receiver(parent=self, port_number=i) for i in range(input_count)]
        self._output_ports = [_pipeline.Sender(parent=self, port_number=i) for i in range(output_count)]

    def get_needed_columns(self):
        """
        Stages which need to have columns inserted into the dataframe, should populate the `self._needed_columns`
        dictionary with mapping of column names to `morpheus.common.TypeId`. This will ensure that the columns are
        allocated and populated with null values.
        """
        return self._needed_columns.copy()

    @abstractmethod
    def compute_schema(self, schema: _pipeline.StageSchema):
        """
        Compute the schema for this stage based on the incoming schema from upstream stages.

        Incoming schema and type information from upstream stages is available via the `schema.input_schemas` and
        `schema.input_types` properties.

        Derived classes need to override this method, can set the output type(s) on `schema` by calling `set_type` for
        all output ports. For example a simple pass-thru stage might perform the following:

        ```
        >>> for (port_idx, port_schema) in enumerate(schema.input_schemas):
        >>>     schema.output_schemas[port_idx].set_type(port_schema.get_type())
        ```

        If the port types in `upstream_schema` are incompatible the stage should raise a `RuntimeError`.
        """
        pass

    def _pre_compute_schema(self, schema: _pipeline.StageSchema):
        """
        Optional pre-flight method, allows base classes like `SinglePortStage` to perform pre-flight checks prior to
        `compute_schema` being called.
        """
        pass
