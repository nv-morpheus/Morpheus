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
import time

import mrc

from morpheus.controllers.rss_controller import RSSController
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module("rss_source", "morpheus")
def rss_source(builder: mrc.Builder):
    """
    A module for applying simple DataFrame schema transform policies.

    This module reads the configuration to determine how to set data types for columns, select, or rename them in the dataframe.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.

    Notes
    -------------
    The configuration should be passed to the module through the `module_config` attribute of the builder. It should
    contain a dictionary where each key is a column name, and the value is another dictionary with keys 'dtype' for
    data type, 'op_type' for operation type ('select' or 'rename'), and optionally 'from' for the original column
    name (if the column is to be renamed).

    Example Configuration
    ---------------------
    {
        "summary": {"dtype": "str", "op_type": "select"},
        "title": {"dtype": "str", "op_type": "select"},
        "content": {"from": "page_content", "dtype": "str", "op_type": "rename"},
        "source": {"from": "link", "dtype": "str", "op_type": "rename"}
    }
    """

    module_config = builder.get_current_module_config()
    rss_config = module_config.get('rss_config', {})

    interval_secs = rss_config.get('interval_secs', 600)
    stop_after = rss_config.get('stop_after', 0)

    # Required configuration keys for RSSController
    required_config_keys = ['feed_input', 'run_indefinitely', 'batch_size',
                            'enable_cache', 'cache_dir', 'cooldown_interval', 'request_timeout']

    # Validate that all required configuration keys are present
    missing_keys = [key for key in required_config_keys if key not in rss_config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Initialize RSSController with validated configuration
    controller = RSSController(
        feed_input=rss_config['feed_input'],
        run_indefinitely=rss_config['run_indefinitely'],
        batch_size=rss_config['batch_size'],
        enable_cache=rss_config['enable_cache'],
        cache_dir=rss_config['cache_dir'],
        cooldown_interval=rss_config['cooldown_interval'],
        request_timeout=rss_config['request_timeout']
    )

    stop_requested = False

    def fetch_feeds() -> MessageMeta:
        """
        Fetch RSS feed entries and yield as MessageMeta object.
        """
        nonlocal stop_requested
        records_emitted = 0

        while (not stop_requested):
            try:
                for df in controller.fetch_dataframes():
                    df_size = len(df)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.info("Received %d new entries...", df_size)
                        logger.info("Emitted %d records so far.", records_emitted)

                    yield MessageMeta(df=df)

                    records_emitted += df_size

                    if (0 < stop_after <= records_emitted):
                        stop_requested = True
                        logger.info("Stop limit reached... preparing to halt the source.")
                        break

            except Exception as exc:
                if not controller.run_indefinitely:
                    logger.error("Failed either in the process of fetching or processing entries: %d.", exc)
                    raise

            if not controller.run_indefinitely:
                stop_requested = True
                continue

            logger.info("Waiting for %d seconds before fetching again...", interval_secs)
            time.sleep(interval_secs)

        logger.info("RSS source exhausted, stopping.")

    node = builder.make_source("fetch_feeds", fetch_feeds)

    builder.register_module_output("output", node)
