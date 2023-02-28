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
import dfp.modules.dfp_split_users  # noqa: F401
import dfp.modules.dfp_training  # noqa: F401
import mrc

import morpheus.modules.file_batcher  # noqa: F401
import morpheus.modules.file_to_df  # noqa: F401
import morpheus.modules.mlflow_model_writer  # noqa: F401
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_config_with_overrides
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DATA_PREP
from ..utils.module_ids import DFP_ROLLING_WINDOW
from ..utils.module_ids import DFP_SPLIT_USERS
from ..utils.module_ids import DFP_TRAINING
from ..utils.module_ids import DFP_TRAINING_PIPELINE

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_TRAINING_PIPELINE, MODULE_NAMESPACE)
def dfp_training_pipeline(builder: mrc.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline modules relevent to training
    process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_TRAINING_PIPELINE, builder)
    config["module_id"] = DFP_TRAINING_PIPELINE
    config["namespace"] = MODULE_NAMESPACE
    config["module_name"] = "dfp_training_pipeline"

    file_batcher_conf = get_config_with_overrides(config, FILE_BATCHER, "file_batcher")
    file_to_df_conf = get_config_with_overrides(config, FILE_TO_DF, "file_to_df")
    dfp_split_users_conf = get_config_with_overrides(config, DFP_SPLIT_USERS, "dfp_split_users")
    dfp_rolling_window_conf = get_config_with_overrides(config, DFP_ROLLING_WINDOW, "dfp_rolling_window")
    dfp_data_prep_conf = get_config_with_overrides(config, DFP_DATA_PREP, "dfp_data_prep")
    dfp_training_conf = get_config_with_overrides(config, DFP_TRAINING, "dfp_training")
    mlflow_model_writer_conf = get_config_with_overrides(config, MLFLOW_MODEL_WRITER, "mlflow_model_writer")

    # Load modules
    file_batcher_module = load_module(file_batcher_conf, builder=builder)
    file_to_dataframe_module = load_module(file_to_df_conf, builder=builder)
    dfp_split_users_modules = load_module(dfp_split_users_conf, builder=builder)
    dfp_rolling_window_module = load_module(dfp_rolling_window_conf, builder=builder)
    dfp_data_prep_module = load_module(dfp_data_prep_conf, builder=builder)
    dfp_training_module = load_module(dfp_training_conf, builder=builder)
    mlflow_model_writer_module = load_module(mlflow_model_writer_conf, builder=builder)

    # Make an edge between the modules.
    builder.make_edge(file_batcher_module.output_port("output"), file_to_dataframe_module.input_port("input"))
    builder.make_edge(file_to_dataframe_module.output_port("output"), dfp_split_users_modules.input_port("input"))
    builder.make_edge(dfp_split_users_modules.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_training_module.input_port("input"))
    builder.make_edge(dfp_training_module.output_port("output"), mlflow_model_writer_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", file_batcher_module.input_port("input"))
    builder.register_module_output("output", mlflow_model_writer_module.output_port("output"))
