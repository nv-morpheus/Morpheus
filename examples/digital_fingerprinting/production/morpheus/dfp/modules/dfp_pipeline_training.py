# Copyright (c) 2022, NVIDIA CORPORATION.
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

import dfp.modules.dfp_training  # noqa: F401
import srf

import morpheus.modules.mlflow_model_writer  # noqa: F401
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(f"morpheus.{__name__}")


@register_module("DFPPipelineTraining", "morpheus_modules")
def dfp_pipeline_training(builder: srf.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline training modules into a single module.

    Parameters
    ----------
    builder : srf.Builder
        Pipeline budler instance.
    """

    module_id = "DFPPipelineTraining"

    config = get_module_config(module_id, builder)

    dfp_training_conf = config.get("DFPTraining", None)
    mlflow_model_writer_conf = config.get("MLFlowModelWriter", None)

    dfp_training_module = load_module(dfp_training_conf, builder=builder)
    mlflow_model_writer_module = load_module(mlflow_model_writer_conf, builder=builder)

    # Make edge between the modules.
    builder.make_edge(dfp_training_module.output_port("output"), mlflow_model_writer_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", dfp_training_module.input_port("input"))
    builder.register_module_output("output", mlflow_model_writer_module.output_port("output"))
