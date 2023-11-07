# Copyright (c) 2023, NVIDIA CORPORATION.
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

import mrc

from morpheus.pipeline.single_port_stage import SinglePortStage


class MyCompoundOpModuleWrapper(SinglePortStage):
    # ... stage implementation
    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        module_config = {}

        my_module = builder.load_module("my_compound_module",
                                        "my_module_namespace",
                                        "module_instance_name",
                                        module_config)

        module_in_node = my_module.input_port("input_0")
        module_out_node = my_module.output_port("output_0")

        builder.make_edge(input_node, module_in_node)

        return module_out_node
