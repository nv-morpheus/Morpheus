# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import mrc
from mrc.core import operators as ops

from _lib import morpheus_example as morpheus_example_cpp
from morpheus.cli.register_stage import register_stage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("pass-thru")
class PassThruStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "pass-thru"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return True

    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        if self._build_cpp_node():
            node = morpheus_example_cpp.PassThruStage(builder, self.unique_name)
        else:
            node = builder.make_node(self.unique_name, ops.map(self.on_data))

        builder.make_edge(input_stream[0], node)
        return node, input_stream[1]
