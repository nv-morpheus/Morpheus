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
    This module function consolidates multiple dfp pipeline modules relevant to the inference process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline builder instance.

    Notes
    ----------
    Configurable parameters:
        - batching_options (dict): Options for batching data, including start and end times, sampling rate, and other settings.
        - cache_dir (str): Directory for caching rolling window data.
        - detection_criteria (dict): Criteria for filtering detections, such as threshold and field_name.
        - inference_options (dict): Options for the inference module, including model settings and other configurations.
        - num_output_ports (int): Number of output ports for the module.
        - preprocessing_options (dict): Options for preprocessing data, including schema and timestamp column name.
        - stream_aggregation_options (dict): Options for aggregating data by stream, including aggregation span and cache settings.
        - timestamp_column_name (str): Name of the timestamp column in the input data.
        - user_splitting_options (dict): Options for splitting data by user, including filtering and user ID column name.
        - write_to_file_options (dict): Options for writing detections to a file, such as filename and overwrite settings.
    """

    config = builder.get_current_module_config()

    cache_dir = config.get("cache_dir")
    ts_column_name = config.get("timestamp_column_name")

    preproc_options = {
        "batching_options": config.get("batching_options", {}),
        "cache_dir": cache_dir,
        "pre_filter_options": {
            "enable_task_filtering": True, "filter_task_type": "inference"
        },
        "timestamp_column_name": ts_column_name,
        "user_splitting_options": config.get("user_splitting_options", {}),
    }

    stream_aggregation_options = config.get("stream_aggregation_options", {})
    stream_aggregation_options = merge_dictionaries(stream_aggregation_options, {
        "cache_dir": cache_dir,
        "timestamp_column_name": ts_column_name,
    })

    data_prep_options = config.get("preprocessing_options", {})
    data_prep_options = merge_dictionaries(data_prep_options, {
        "timestamp_column_name": ts_column_name,
    })

    inference_model_options = config.get("inference_options", {})

    detection_criteria = config.get("detection_criteria", {})

    post_processing_options = {
        "timestamp_column_name": ts_column_name,
    }

    serialize_options = config.get("serialize_options", {})

    write_to_file_options = config.get("write_to_file_options", {})

    preproc_defaults = {}  # placeholder for future defaults
    preproc_conf = merge_dictionaries(preproc_options, preproc_defaults)

    stream_aggregation_defaults = {
        "cache_mode": "batch",
        "trigger_on_min_history": 300,
        "trigger_on_min_increment": 300,
    }
    dfp_rolling_window_conf = merge_dictionaries(stream_aggregation_options, stream_aggregation_defaults)

    data_prep_defaults = {}  # placeholder for future defaults
    dfp_data_prep_conf = merge_dictionaries(data_prep_options, data_prep_defaults)

    inference_model_defaults = {}  # placeholder for future defaults
    dfp_inference_conf = merge_dictionaries(inference_model_options, inference_model_defaults)

    detection_criteria_defaults = {"field_name": "mean_abs_z", "threshold": 2.0, "filter_source": "DATAFRAME"}
    filter_detections_conf = merge_dictionaries(detection_criteria, detection_criteria_defaults)

    post_processing_defaults = {}  # placeholder for future defaults
    dfp_post_proc_conf = merge_dictionaries(post_processing_options, post_processing_defaults)

    serialize_defaults = {"exclude": ['batch_count', 'origin_hash', '_row_hash', '_batch_id'], "use_cpp": True}
    serialize_conf = merge_dictionaries(serialize_options, serialize_defaults)

    write_to_file_defaults = {
        "filename": "dfp_inference_output.csv",
    }
    write_to_file_conf = merge_dictionaries(write_to_file_options, write_to_file_defaults)

    # Load modules
    preproc_module = builder.load_module(DFP_PREPROC, "morpheus", "dfp_preproc", preproc_conf)
    dfp_rolling_window_module = builder.load_module(DFP_ROLLING_WINDOW,
                                                    "morpheus",
                                                    "dfp_rolling_window",
                                                    dfp_rolling_window_conf)
    dfp_data_prep_module = builder.load_module(DFP_DATA_PREP, "morpheus", "dfp_data_prep", dfp_data_prep_conf)
    dfp_inference_module = builder.load_module(DFP_INFERENCE, "morpheus", "dfp_inference", dfp_inference_conf)
    filter_detections_module = builder.load_module(FILTER_DETECTIONS,
                                                   "morpheus",
                                                   "filter_detections",
                                                   filter_detections_conf)
    dfp_post_proc_module = builder.load_module(DFP_POST_PROCESSING,
                                               "morpheus",
                                               "dfp_post_processing",
                                               dfp_post_proc_conf)
    serialize_module = builder.load_module(SERIALIZE, "morpheus", "serialize", serialize_conf)
    write_to_file_module = builder.load_module(WRITE_TO_FILE, "morpheus", "write_to_file", write_to_file_conf)

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
