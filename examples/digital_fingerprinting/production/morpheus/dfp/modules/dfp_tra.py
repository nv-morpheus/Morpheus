# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import dfp.modules.dfp_data_prep  # noqa: F401
import dfp.modules.dfp_rolling_window  # noqa: F401
import dfp.modules.dfp_training  # noqa: F401
import mrc

import morpheus.modules.mlflow_model_writer  # noqa: F401
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DATA_PREP
from ..utils.module_ids import DFP_ROLLING_WINDOW
from ..utils.module_ids import DFP_TRA
from ..utils.module_ids import DFP_TRAINING

logger = logging.getLogger(__name__)


@register_module(DFP_TRA, MODULE_NAMESPACE)
def dfp_tra(builder: mrc.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline modules relevent to training
    process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_TRA, builder)

    dfp_rolling_window_conf = config.get(DFP_ROLLING_WINDOW, None)
    dfp_data_prep_conf = config.get(DFP_DATA_PREP, None)
    dfp_training_conf = config.get(DFP_TRAINING, None)
    mlflow_model_writer_conf = config.get(MLFLOW_MODEL_WRITER, None)

    # Load modules
    dfp_rolling_window_module = load_module(dfp_rolling_window_conf, builder=builder)
    dfp_data_prep_module = load_module(dfp_data_prep_conf, builder=builder)
    dfp_training_module = load_module(dfp_training_conf, builder=builder)
    mlflow_model_writer_module = load_module(mlflow_model_writer_conf, builder=builder)

    # Make an edge between the modules.
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_training_module.input_port("input"))
    builder.make_edge(dfp_training_module.output_port("output"), mlflow_model_writer_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", dfp_rolling_window_module.input_port("input"))
    builder.register_module_output("output", mlflow_model_writer_module.output_port("output"))
