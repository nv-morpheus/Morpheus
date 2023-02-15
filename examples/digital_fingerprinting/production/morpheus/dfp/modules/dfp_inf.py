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
import dfp.modules.dfp_rolling_window  # noqa: F401
import dfp.modules.dfp_split_users  # noqa: F401
import mrc

import morpheus.modules.filter_detections  # noqa: F401
import morpheus.modules.serialize  # noqa: F401
import morpheus.modules.write_to_file  # noqa: F401
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_ids import WRITE_TO_FILE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DATA_PREP
from ..utils.module_ids import DFP_INF
from ..utils.module_ids import DFP_INFERENCE
from ..utils.module_ids import DFP_POST_PROCESSING
from ..utils.module_ids import DFP_ROLLING_WINDOW

logger = logging.getLogger(__name__)


@register_module(DFP_INF, MODULE_NAMESPACE)
def dfp_inf(builder: mrc.Builder):
    """
    This module function allows for the consolidation of multiple dfp pipeline modules relevent to inference
    process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_INF, builder)

    dfp_rolling_window_conf = config.get(DFP_ROLLING_WINDOW, None)
    dfp_data_prep_conf = config.get(DFP_DATA_PREP, None)
    dfp_inference_conf = config.get(DFP_INFERENCE, None)
    filter_detections_conf = config.get(FILTER_DETECTIONS, None)
    dfp_post_proc_conf = config.get(DFP_POST_PROCESSING, None)
    serialize_conf = config.get(SERIALIZE, None)
    write_to_file_conf = config.get(WRITE_TO_FILE, None)

    # Load modules
    dfp_rolling_window_module = load_module(dfp_rolling_window_conf, builder=builder)
    dfp_data_prep_module = load_module(dfp_data_prep_conf, builder=builder)
    dfp_inference_module = load_module(dfp_inference_conf, builder=builder)
    filter_detections_module = load_module(filter_detections_conf, builder=builder)
    dfp_post_proc_module = load_module(dfp_post_proc_conf, builder=builder)
    serialize_module = load_module(serialize_conf, builder=builder)
    write_to_file_module = load_module(write_to_file_conf, builder=builder)

    # Make an edge between the modules.
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_inference_module.input_port("input"))
    builder.make_edge(dfp_inference_module.output_port("output"), filter_detections_module.input_port("input"))
    builder.make_edge(filter_detections_module.output_port("output"), dfp_post_proc_module.input_port("input"))
    builder.make_edge(dfp_post_proc_module.output_port("output"), serialize_module.input_port("input"))
    builder.make_edge(serialize_module.output_port("output"), write_to_file_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", dfp_rolling_window_module.input_port("input"))
    builder.register_module_output("output", write_to_file_module.output_port("output"))
