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
import pickle
from functools import partial

import mrc
from dfp.utils.module_ids import DFP_MONITOR
from mrc.core import operators as ops
from tqdm import tqdm

from morpheus.controllers.monitor_controller import MonitorController
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module
from morpheus.utils.monitor_utils import MorpheusTqdm
from morpheus.utils.monitor_utils import SilentMorpheusTqdm

logger = logging.getLogger(f"morpheus.{__name__}")


@register_module(DFP_MONITOR, MORPHEUS_MODULE_NAMESPACE)
def monitor(builder: mrc.Builder):
    """
    This module function is used for monitoring pipeline message rate.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline builder instance.

    Notes
    -----
    Configurable parameters:
        - description (str): Name to show for this Monitor Stage in the console window; Example: 'Progress';
        Default: 'Progress'
        - silence_monitors (bool): Slience the monitors on the console; Example: True; Default: False
        - smoothing (float): Smoothing parameter to determine how much the throughput should be averaged.
        0 = Instantaneous, 1 = Average.; Example: 0.01; Default: 0.05
        - unit (str): Units to show in the rate value.; Example: 'messages'; Default: 'messages'
        - delayed_start (bool): When delayed_start is enabled, the progress bar will not be shown until the first
        message is received. Otherwise, the progress bar is shown on pipeline startup and will begin timing
        immediately. In large pipelines, this option may be desired to give a more accurate timing;
        Example: True; Default: False
        - determine_count_fn_schema (str): Custom function for determining the count in a message. Gets called for
        each message. Allows for correct counting of batched and sliced messages.; Example: func_str; Default: None
        - log_level (str): Enable this stage when the configured log level is at `log_level` or lower;
        Example: 'DEBUG'; Default: INFO
    """

    config = builder.get_current_module_config()

    description = config.get("description", "Progress")
    silence_monitors = config.get("silence_monitors", False)
    smoothing = config.get("smoothing", 0.05)
    unit = config.get("unit", "messages")
    delayed_start = config.get("delayed_start", False)
    determine_count_fn_schema = config.get("determine_count_fn_schema", None)
    log_level = config.get("log_level", "INFO")

    log_level = logging.getLevelName(log_level)

    determine_count_fn = None
    if determine_count_fn_schema is not None:
        if hasattr(determine_count_fn_schema, "schema_str") and hasattr(determine_count_fn_schema, "encoding"):
            determine_count_fn = pickle.loads(
                bytes(determine_count_fn_schema.get("schema_str"), determine_count_fn_schema.get("encoding")))

    if silence_monitors:
        tqdm_class = SilentMorpheusTqdm
    else:
        tqdm_class = MorpheusTqdm

    position = MonitorController.controller_count
    controller = MonitorController(position=position,
                                   description=description,
                                   smoothing=smoothing,
                                   unit=unit,
                                   delayed_start=delayed_start,
                                   determine_count_fn=determine_count_fn,
                                   log_level=log_level,
                                   tqdm_class=tqdm_class)

    MonitorController.controller_count += 1

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(controller.progress_sink),
                 ops.on_completed(partial(controller.sink_on_completed, tqdm_class=tqdm_class))).subscribe(sub)

    if controller.is_enabled():
        # Set the monitor interval to 0 to use prevent using tqdms monitor
        tqdm.monitor_interval = 0

    # Start the progress bar if we dont have a delayed start
    if (not controller.delayed_start):
        controller.ensure_progress_bar()

    node = builder.make_node(DFP_MONITOR, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
