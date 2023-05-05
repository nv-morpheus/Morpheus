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
import queue
import time

import mrc

from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import MULTIPLEXER
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(MULTIPLEXER, MORPHEUS_MODULE_NAMESPACE)
def multiplexer(builder: mrc.Builder):
    """
    The multiplexer receives data packets from one or more input ports and interleaves them into a single output.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - num_input_ports_to_merge (int): Number of nodes stream data to be combined; Example: `3`;
            Default: `2`
            - stop_after_secs (int): Time in seconds to halt the process; Example: `10`;
            Default: -1 (runs indefinitely).
            - streaming (bool): Execution in streaming mode is indicated by this flag; Example: `True`;
            Default: False
    """

    config = builder.get_current_module_config()

    num_input_ports_to_merge = config.get("num_input_ports_to_merge", 2)
    streaming = config.get("streaming", False)
    stop_after_secs = config.get("stop_after_secs", -1)

    if num_input_ports_to_merge <= 0:
        raise ValueError("The value for the 'num_input_ports_to_merge' must be > 0")

    q = queue.Queue()

    def on_next(data):
        nonlocal q
        q.put(data)

    def on_error():
        pass

    def on_complete():
        pass

    def read_from_q():
        nonlocal q
        start_time = time.monotonic()  # Get the start time in monotonic seconds
        while True:
            try:
                # Try to get the next item from the queue, waiting for up to 1 second
                yield q.get(block=True, timeout=1.0)
            except queue.Empty:
                # If the queue is empty and we're not in streaming mode, wait a short amount of time and break the loop
                if not streaming:
                    time.sleep(0.01)
                    break

            # Calculate the elapsed time in monotonic seconds
            elapsed_time = time.monotonic() - start_time

            # If we've been running for longer than the stop_after_secs value (if it's set), break the loop
            if stop_after_secs > 0 and elapsed_time >= stop_after_secs:
                break

    for i in range(num_input_ports_to_merge):
        in_port = f"input-{i}"
        in_node = builder.make_sink(in_port, on_next, on_error, on_complete)
        # Register input port for a module.
        builder.register_module_input(in_port, in_node)

    out_node = builder.make_source("output", read_from_q)

    # Register output port for a module.
    builder.register_module_output("output", out_node)
