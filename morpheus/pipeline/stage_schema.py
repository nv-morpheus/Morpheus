# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

if typing.TYPE_CHECKING:
    from .stream_wrapper import StreamWrapper


class PortSchema:

    def __init__(self, port_type: type = None):
        self._type = port_type
        self._completed = False

    def get_type(self) -> type:
        return self._type

    def set_type(self, value: type):
        assert not self._completed, "Attempted to set type on completed PortSchema"
        self._type = value

    def complete(self):
        assert self._type is not None, "Attempted to complete PortSchema without setting type"
        self._completed = True

    def is_completed(self) -> bool:
        return self._completed


class StageSchema:

    def __init__(self, stage: "StreamWrapper"):
        self._stage = stage  # TODO: Determine if we need to hold a reference to the stage

        self._input_schemas = []
        for port in stage.input_ports:
            assert port._schema.is_completed(), "Attempted to create StageSchema with incomplete input port schemas"
            self._input_schemas.append(port._schema)

        self._output_schemas = [PortSchema() for _ in range(len(stage.output_ports))]

    @property
    def input_schemas(self) -> list[PortSchema]:
        return self._input_schemas

    @property
    def input_schema(self) -> PortSchema:
        """
        Single port variant of input_schemas. Will fail if there are multiple input ports.
        """
        assert len(self._input_schemas) == 1, \
            "Attempted to access input_schema property on StageSchema with multiple inputs"
        return self._input_schemas[0]

    @property
    def output_schemas(self) -> list[PortSchema]:
        return self._output_schemas

    @property
    def output_schema(self) -> PortSchema:
        """
        Single port variant of output_schemas. Will fail if there are multiple output ports.
        """
        assert len(self._output_schemas) == 1, \
            "Attempted to access output_schema property on StageSchema with multiple outputs"
        return self._output_schemas[0]

    def complete(self):
        for port_schema in self.output_schemas:
            # This locks the port schema
            port_schema.complete()
