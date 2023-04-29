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
from functools import reduce

import dfp.utils.monitor_utils as mu
import mrc
from dfp.utils.module_ids import DFP_MONITOR
from mrc.core import operators as ops
from tqdm import tqdm

import cudf

from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.stages.general.monitor_stage import MorpheusTqdm
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(f"morpheus.{__name__}")

MONITOR_COUNT = 0


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
        - description (string): Name to show for this Monitor Stage in the console window; Example: 'Progress';
        Default: 'Progress'
        - silence_monitors (bool): Slience the monitors on the console; Example: True; Default: False
        - smoothing (float): Smoothing parameter to determine how much the throughput should be averaged.
        0 = Instantaneous, 1 = Average.; Example: 0.01; Default: 0.05
        - unit (string): Units to show in the rate value.; Example: 'messages'; Default: 'messages'
        - delayed_start (bool): When delayed_start is enabled, the progress bar will not be shown until the first
        message is received. Otherwise, the progress bar is shown on pipeline startup and will begin timing
        immediately. In large pipelines, this option may be desired to give a more accurate timing;
        Example: True; Default: False
        - determine_count_fn_schema (string): Custom function for determining the count in a message. Gets called for
        each message. Allows for correct counting of batched and sliced messages.; Example: func_str; Default: None
        - log_level (string): Enable this stage when the configured log level is at `log_level` or lower;
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

    progress = None

    global MONITOR_COUNT
    position = MONITOR_COUNT
    MONITOR_COUNT += 1
    enabled = None

    log_level = logging.getLevelName(log_level)

    determine_count_fn = None
    if determine_count_fn_schema is not None:
        if hasattr(determine_count_fn_schema, "schema_str") and hasattr(determine_count_fn_schema, "encoding"):
            determine_count_fn = pickle.loads(
                bytes(determine_count_fn_schema.get("schema_str"), determine_count_fn_schema.get("encoding")))

    def _is_enabled() -> bool:
        nonlocal enabled
        if enabled is None:
            enabled = logger.isEnabledFor(log_level)

        return enabled

    def _auto_count_fn(x):

        if (x is None):
            return None

        # Wait for a list thats not empty
        if (isinstance(x, list) and len(x) == 0):
            return None

        if (isinstance(x, cudf.DataFrame)):
            return lambda y: len(y.index)
        elif (isinstance(x, MultiMessage)):
            return lambda y: y.mess_count
        elif (isinstance(x, MessageMeta)):
            return lambda y: y.count
        elif isinstance(x, ControlMessage):

            def check_df(y):
                df = y.payload().df
                if df is not None:
                    return len(df)
                else:
                    return 0

            return check_df
        elif (isinstance(x, list)):
            item_count_fn = _auto_count_fn(x[0])
            return lambda y: reduce(lambda sum, z, item_count_fn=item_count_fn: sum + item_count_fn(z), y, 0)
        elif (isinstance(x, str)):
            return lambda y: 1
        elif (hasattr(x, "__len__")):
            return len  # Return len directly (same as `lambda y: len(y)`)
        else:
            raise NotImplementedError(f"Unsupported type: {type(x)}")

    def _ensure_progress_bar():
        nonlocal progress
        if (progress is None):
            progress = tqdm_class(desc=description,
                                  smoothing=smoothing,
                                  dynamic_ncols=True,
                                  unit=(unit if unit.startswith(" ") else f" {unit}"),
                                  mininterval=0.25,
                                  maxinterval=1.0,
                                  miniters=1,
                                  position=position)

            progress.reset()

    def progress_sink(x):

        # Make sure the progress bar is shown
        _ensure_progress_bar()

        nonlocal determine_count_fn
        if (determine_count_fn is None):
            determine_count_fn = _auto_count_fn(x)

        # Skip incase we have empty objects
        if (determine_count_fn is None):
            return x

        # Do our best to determine the count
        n = determine_count_fn(x)
        progress.update(n=n)

        return x

    def sink_on_completed():
        progress.set_description_str(progress.desc + "[Complete]")

        progress.stop()

        # To prevent the monitors from writing over eachother, stop the monitor when the last stage completes
        global MONITOR_COUNT
        MONITOR_COUNT -= 1

        if (MONITOR_COUNT <= 0 and tqdm_class.monitor is not None):
            tqdm_class.monitor.exit()
            tqdm_class.monitor = None

    if silence_monitors:
        tqdm_class = mu.SilentMorpheusTqdm
    else:
        tqdm_class = MorpheusTqdm

    if _is_enabled():
        # Set the monitor interval to 0 to use prevent using tqdms monitor
        tqdm.monitor_interval = 0

        # Start the progress bar if we dont have a delayed start
    if (not delayed_start):
        _ensure_progress_bar()

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(progress_sink), ops.on_completed(sink_on_completed)).subscribe(sub)

    node = builder.make_node(DFP_MONITOR, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
