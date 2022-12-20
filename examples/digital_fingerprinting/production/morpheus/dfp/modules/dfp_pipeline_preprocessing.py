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

import dfp.modules.dfp_preprocessing  # noqa: F401
import dfp.modules.dfp_rolling_window  # noqa: F401
import dfp.modules.dfp_split_users  # noqa: F401
import dfp.modules.dfp_training  # noqa: F401
import mrc

import morpheus.modules.file_batcher  # noqa: F401
import morpheus.modules.file_to_df  # noqa: F401
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(f"morpheus.{__name__}")


@register_module("DFPPipelinePreprocessing", "morpheus_modules")
def dfp_pipeline_preprocessing(builder: mrc.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline preprocessing modules
    into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    module_id = "DFPPipelinePreprocessing"

    config = get_module_config(module_id, builder)

    file_batcher_conf = config.get("FileBatcher", None)
    file_to_df_conf = config.get("FileToDataFrame", None)
    dfp_split_users_conf = config.get("DFPSplitUsers", None)
    dfp_rolling_window_conf = config.get("DFPRollingWindow", None)
    dfp_preprocessing_conf = config.get("DFPPreprocessing", None)

    # Load modules
    file_batcher_module = load_module(file_batcher_conf, builder=builder)
    file_to_dataframe_module = load_module(file_to_df_conf, builder=builder)
    dfp_split_users_modules = load_module(dfp_split_users_conf, builder=builder)
    dfp_rolling_window_module = load_module(dfp_rolling_window_conf, builder=builder)
    dfp_preprocessing_module = load_module(dfp_preprocessing_conf, builder=builder)

    # Make edge between the modules.
    builder.make_edge(file_batcher_module.output_port("output"), file_to_dataframe_module.input_port("input"))
    builder.make_edge(file_to_dataframe_module.output_port("output"), dfp_split_users_modules.input_port("input"))
    builder.make_edge(dfp_split_users_modules.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_preprocessing_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", file_batcher_module.input_port("input"))
    builder.register_module_output("output", dfp_preprocessing_module.output_port("output"))
