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

import logging

import dfp.modules.dfp_inf  # noqa: F401
import dfp.modules.dfp_preproc  # noqa: F401
import dfp.modules.dfp_tra  # noqa: F401
import mrc
from mrc.core.node import Broadcast

from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DEPLOYMENT
from ..utils.module_ids import DFP_INF
from ..utils.module_ids import DFP_PREPROC
from ..utils.module_ids import DFP_TRA

logger = logging.getLogger(__name__)


@register_module(DFP_DEPLOYMENT, MODULE_NAMESPACE)
def dfp_inf(builder: mrc.Builder):

    module_config = get_module_config(DFP_DEPLOYMENT, builder)

    preproc_conf = module_config.get(DFP_PREPROC, None)
    infer_conf = module_config.get(DFP_INF, None)
    train_conf = module_config.get(DFP_TRA, None)

    if "output_port_count" not in module_config:
        raise Exception("Missing required attribute 'output_port_count'")

    output_port_count = module_config.get("output_port_count")

    preproc_module = load_module(preproc_conf, builder=builder)

    if (train_conf is not None and infer_conf is not None):

        # Load module from registry.
        infer_module = load_module(infer_conf, builder=builder)
        train_module = load_module(train_conf, builder=builder)

        # Create broadcast node to fork the pipeline.
        boradcast_node = Broadcast(builder, "broadcast")

        # Make an edge between modules
        builder.make_edge(preproc_module.output_port("output"), boradcast_node)
        builder.make_edge(boradcast_node, infer_module.input_port("input"))
        builder.make_edge(boradcast_node, train_module.input_port("input"))

        out_streams = [train_module.output_port("output"), infer_module.output_port("output")]

    elif infer_conf is not None:
        infer_module = load_module(infer_conf, builder=builder)
        builder.make_edge(preproc_module.output_port("output"), infer_module.input_port("input"))
        out_streams = [infer_module.output_port("output")]

    elif train_conf is not None:
        train_module = load_module(train_conf, builder=builder)
        builder.make_edge(preproc_module.output_port("output"), train_module.input_port("input"))
        out_streams = [train_module.output_port("output")]

    else:
        raise Exception("Expected DFP deployment workload_types are not found.")

    # Register input port for a module.
    builder.register_module_input("input", preproc_module.input_port("input"))

    # Register output ports for a module.
    for i in range(output_port_count):
        # Output ports are registered in increment order.
        builder.register_module_output(f"output-{i}", out_streams[i])
