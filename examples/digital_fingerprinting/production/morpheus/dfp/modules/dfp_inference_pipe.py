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
import dfp.modules.dfp_inference  # noqa: F401
import dfp.modules.dfp_postprocessing  # noqa: F401
import dfp.modules.dfp_preproc  # noqa: F401
import dfp.modules.dfp_rolling_window
import mrc

import morpheus.modules.filter_detections  # noqa: F401
import morpheus.modules.serialize  # noqa: F401
import morpheus.modules.write_to_file  # noqa: F401
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_ids import WRITE_TO_FILE
from morpheus.utils.module_utils import merge_dictionaries
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DATA_PREP
from ..utils.module_ids import DFP_INFERENCE
from ..utils.module_ids import DFP_INFERENCE_PIPE
from ..utils.module_ids import DFP_POST_PROCESSING
from ..utils.module_ids import DFP_PREPROC
from ..utils.module_ids import DFP_ROLLING_WINDOW

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_INFERENCE_PIPE, MORPHEUS_MODULE_NAMESPACE)
def dfp_inference_pipe(builder: mrc.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline modules relevent to inference
    process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = builder.get_current_module_config()

    cache_dir = config.get("cache_dir")
    ts_column_name = config.get("timestamp_column_name")

    preproc_options = {
        "batching_options": config.get("batching_options", {}),
        "cache_dir": cache_dir,
        "pre_filter_options": {
            "enable_task_filtering": True,
            "filter_task_type": "inference"
        },
        "timestamp_column_name": ts_column_name,
        "user_splitting_options": config.get("user_splitting_options", {}),
    }

    preproc_defaults = {}
    preproc_conf = merge_dictionaries(preproc_options, preproc_defaults)

    dfp_rolling_window_conf = config[DFP_ROLLING_WINDOW]
    dfp_data_prep_conf = config[DFP_DATA_PREP]
    dfp_inference_conf = config[DFP_INFERENCE]
    filter_detections_conf = config[FILTER_DETECTIONS]
    dfp_post_proc_conf = config[DFP_POST_PROCESSING]
    serialize_conf = config[SERIALIZE]
    write_to_file_conf = config[WRITE_TO_FILE]

    # Load modules
    preproc_module = builder.load_module(DFP_PREPROC, "morpheus", "dfp_preproc", preproc_conf)
    dfp_rolling_window_module = builder.load_module(DFP_ROLLING_WINDOW, "morpheus", "dfp_rolling_window",
                                                    dfp_rolling_window_conf)
    dfp_data_prep_module = builder.load_module(DFP_DATA_PREP, "morpheus", "dfp_data_prep", dfp_data_prep_conf)
    dfp_inference_module = builder.load_module(DFP_INFERENCE, "morpheus", "dfp_inference", dfp_inference_conf)
    filter_detections_module = builder.load_module(FILTER_DETECTIONS, "morpheus", "filter_detections",
                                                   filter_detections_conf)
    dfp_post_proc_module = builder.load_module(DFP_POST_PROCESSING, "morpheus", "dfp_post_processing",
                                               dfp_post_proc_conf)
    serialize_module = builder.load_module(SERIALIZE, "morpheus", "serialize", serialize_conf)
    write_to_file_module = builder.load_module(WRITE_TO_FILE, "morpheus", "write_to_file",
                                               write_to_file_conf)

    # Make an edge between the modules.
    builder.make_edge(preproc_module.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_inference_module.input_port("input"))
    builder.make_edge(dfp_inference_module.output_port("output"), filter_detections_module.input_port("input"))
    builder.make_edge(filter_detections_module.output_port("output"), dfp_post_proc_module.input_port("input"))
    builder.make_edge(dfp_post_proc_module.output_port("output"), serialize_module.input_port("input"))
    builder.make_edge(serialize_module.output_port("output"), write_to_file_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", preproc_module.input_port("input"))
    builder.register_module_output("output", write_to_file_module.output_port("output"))
