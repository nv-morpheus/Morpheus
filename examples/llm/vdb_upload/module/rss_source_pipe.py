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

import mrc

from morpheus.modules.general.monitor import Monitor
from morpheus.modules.input.rss_source import rss_source  # noqa: F401
from morpheus.modules.preprocess.deserialize import deserialize  # noqa: F401
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module
from .schema_transform import schema_transform  # noqa: F401
from ...common.web_scraper_module import web_scraper  # noqa: F401

logger = logging.getLogger(__name__)


@register_module("rss_source_pipe", "morpheus_examples_llm")
def _rss_source_pipe(builder: mrc.Builder):
    """
    Creates a pipeline for processing RSS feeds.

    This function sets up a pipeline that takes RSS feed data, scrapes web content
    based on the feed, and then transforms the scraped data according to a specified schema.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder to which the pipeline modules will be added.
    """

    # Load the module configuration from the builder
    module_config = builder.get_current_module_config()
    rss_config = module_config.get("rss_config", {})
    enable_monitor = rss_config.get("enable_monitor", False)

    # Build sub-module configurations
    rss_source_config = {
        "module_id": "rss_source",
        "module_name": "rss_source",
        "namespace": "morpheus",
        "rss_config": rss_config,
    }

    web_scraper_config = {
        "module_id": "web_scraper",
        "module_name": "web_scraper",
        "namespace": "morpheus_examples_llm",
        "web_scraper_config": rss_config.get("web_scraper_config", {}),
    }

    transform_config = {
        "module_id": "schema_transform",
        "module_name": "schema_transform",
        "namespace": "morpheus_examples_llm",
        "schema_transform_config": {
            "summary": {"dtype": "str", "op_type": "select"},
            "title": {"dtype": "str", "op_type": "select"},
            "content": {"from": "page_content", "dtype": "str", "op_type": "rename"},
            "source": {"from": "link", "dtype": "str", "op_type": "rename"}
        }
    }

    deserialize_config = {
        "module_id": "deserialize",
        "module_name": "deserialize",
        "namespace": "morpheus",
        "batch_size": rss_config.get("batch_size", 512),
    }

    monitor_m1 = Monitor.get_definition("monitor_m1", {"description": "RSSSourcePipe RSS Source",
                                                       "silence_monitors": not enable_monitor})
    monitor_0 = Monitor.get_definition("monitor_0", {"description": "RSSSourcePipe Web Scraper",
                                                     "silence_monitors": not enable_monitor})
    monitor_1 = Monitor.get_definition("monitor_1", {"description": "RSSSourcePipe Transform",
                                                     "silence_monitors": not enable_monitor})
    monitor_2 = Monitor.get_definition("monitor_2", {"description": "RSSSourcePipe Deserialize",
                                                     "silence_monitors": not enable_monitor})

    # Load modules
    rss_source_module = load_module(config=rss_source_config, builder=builder)
    monitor_m1 = monitor_m1.load(builder=builder)
    web_scraper_module = load_module(config=web_scraper_config, builder=builder)
    monitor_0_module = monitor_0.load(builder=builder)
    transform_module = load_module(config=transform_config, builder=builder)
    monitor_1_module = monitor_1.load(builder=builder)
    deserialize_module = load_module(config=deserialize_config, builder=builder)
    monitor_2_module = monitor_2.load(builder=builder)

    # Connect the modules: RSS source -> Web scraper -> Schema transform
    builder.make_edge(rss_source_module.output_port("output"), monitor_m1.input_port("input"))
    builder.make_edge(monitor_m1.output_port("output"), web_scraper_module.input_port("input"))
    builder.make_edge(web_scraper_module.output_port("output"), monitor_0_module.input_port("input"))
    builder.make_edge(monitor_0_module.output_port("output"), transform_module.input_port("input"))
    builder.make_edge(transform_module.output_port("output"), monitor_1_module.input_port("input"))
    builder.make_edge(monitor_1_module.output_port("output"), deserialize_module.input_port("input"))
    builder.make_edge(deserialize_module.output_port("output"), monitor_2_module.input_port("input"))

    # Register the final output of the transformation module
    builder.register_module_output("output", monitor_2_module.output_port("output"))


RSSSourcePipe = ModuleInterface("rss_source_pipe", "morpheus_examples_llm")
