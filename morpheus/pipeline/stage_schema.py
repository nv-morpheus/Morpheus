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


class StagePortSchema:

    def __init__(self, port_type: type = None) -> None:
        self._type = port_type
        self._completed = False

    @property
    def type(self) -> type:
        return self._type

    @type.setter
    def type(self, value: type):
        assert not self._completed, "Attempted to set type on completed StagePortSchema"

        self._type = value

    def complete(self):
        assert self.type is not None, "Attempted to complete StagePortSchema without setting type"
        self._completed = True


class StageSchema:

    def __init__(self, stage: "StreamWrapper") -> None:
        self._stage = stage

        self._input_ports = [p._schema for p in stage.input_ports]
        self._output_ports = [StagePortSchema() for p in range(len(stage.output_ports))]

    def complete(self):

        for port in self._output_ports:
            # This locks the port schema
            port.complete()