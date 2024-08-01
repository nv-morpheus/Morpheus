# Copyright (c) 2024, NVIDIA CORPORATION.
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
from pydantic import ValidationError

from morpheus.controllers.rss_controller import RSSController
from morpheus.messages import MessageMeta
from morpheus.modules.schemas.rss_source_schema import RSSSourceSchema
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

RSSSourceLoaderFactory = ModuleLoaderFactory("rss_source", "morpheus", RSSSourceSchema)


@register_module("rss_source", "morpheus")
def _rss_source(builder: mrc.Builder):
    """
    A module for loading RSS feed items into a DataFrame.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.

    Example Configuration
    ---------------------
    {
        "batch_size": 32,
        "cache_dir": "./.cache/http",
        "cooldown_interval_sec": 600,
        "enable_cache": True,
        "feed_input": ["https://nvidianews.nvidia.com/releases.xml"],
        "interval_sec": 600,
        "request_timeout_sec": 2.0,
        run_indefinitely: True,
        "stop_after_rec": 0,
        "strip_markup": True,
    }
    """

    module_config = builder.get_current_module_config()
    rss_config = module_config.get("rss_source", {})
    try:
        validated_config = RSSSourceSchema(**rss_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid RSS source configuration: {error_messages}"
        logger.error(log_error_message)

        raise

    # Initialize RSSController with validated configuration
    controller = RSSController(feed_input=validated_config.feed_input,
                               run_indefinitely=validated_config.run_indefinitely,
                               batch_size=validated_config.batch_size,
                               enable_cache=validated_config.enable_cache,
                               cache_dir=validated_config.cache_dir,
                               cooldown_interval=validated_config.cooldown_interval_sec,
                               request_timeout=validated_config.request_timeout_sec,
                               strip_markup=validated_config.strip_markup)

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

                    if (0 < validated_config.stop_after_rec <= records_emitted):
                        stop_requested = True
                        logger.info("Stop limit reached... preparing to halt the source.")
                        break

            except Exception as exc:
                if not controller.run_indefinitely:
                    logger.error("Failed either in the process of fetching or processing entries: %s.", exc)
                    raise
                logger.error("Failed either in the process of fetching or processing entries: %s.", exc)

            if not controller.run_indefinitely:
                stop_requested = True
                continue

            logger.info("Waiting for %d seconds before fetching again...", validated_config.interval_sec)
            time.sleep(validated_config.interval_sec)

        logger.info("RSS source exhausted, stopping.")

    node = builder.make_source("fetch_feeds", fetch_feeds)

    builder.register_module_output("output", node)
