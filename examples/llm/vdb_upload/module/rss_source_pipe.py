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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import mrc
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import validator

from morpheus.modules.general.monitor import Monitor
from morpheus.modules.input.rss_source import RSSSourceInterface
from morpheus.modules.preprocess.deserialize import DeserializeInterface
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import register_module
from .schema_transform import SchemaTransformInterface
from ...common.web_scraper_module import WebScraperInterface

logger = logging.getLogger(__name__)


class RSSSourceParamContract(BaseModel):
    batch_size: int = 32
    cache_dir: str = "./.cache/http"
    cooldown_interval_sec: int = 600
    enable_cache: bool = False
    enable_monitor: bool = True
    feed_input: List[str] = Field(default_factory=list)
    interval_sec: int = 600
    output_batch_size: int = 2048
    request_timeout_sec: float = 2.0
    run_indefinitely: bool = True
    stop_after: int = 0
    web_scraper_config: Optional[Dict[Any, Any]] = None

    @validator('feed_input', pre=True)
    def validate_feed_input(cls, v):
        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return v
        raise ValueError('feed_input must be a string or a list of strings')


@register_module("rss_source_pipe", "morpheus_examples_llm")
def _rss_source_pipe(builder: mrc.Builder):
    """
    Creates a pipeline for processing RSS feeds.

    This function sets up a pipeline that takes RSS feed data, scrapes web content
    based on the feed, and then outputs the scraped data. It integrates modules like RSS source,
    web scraper, and deserializer, along with monitoring for each stage.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder to which the pipeline modules will be added.

    Notes
    -----
    The module configuration can include the following parameters:

    - **rss_config**: Configuration for the RSS source module.
      - **batch_size**: Number of RSS feed items to process in each batch.
      - **cache_dir**: Directory for caching RSS feed data.
      - **cooldown_interval_sec**: Cooldown interval in seconds between fetches.
      - **enable_cache**: Boolean to enable caching of feed data.
      - **enable_monitor**: Boolean to enable monitoring for this module.
      - **feed_input**: List of RSS feed URLs to process.
      - **interval_sec**: Interval in seconds for fetching new feed items.
      - **request_timeout_sec**: Timeout in seconds for RSS feed requests.
      - **run_indefinitely**: Boolean to indicate continuous running.
      - **stop_after**: Number of records to process before stopping (0 for indefinite).
      - **web_scraper_config**: Configuration for the web scraper module.
        - **chunk_overlap**: Overlap size for chunks in web scraping.
        - **chunk_size**: Size of content chunks for processing.
        - **enable_cache**: Boolean to enable caching of scraped data.

    The pipeline connects these modules in the following order:
    RSS Source -> Web Scraper -> Deserializer, with monitoring at each stage.
    """

    # Load and validate the module configuration from the builder
    module_config = builder.get_current_module_config()
    rss_config = module_config.get("rss_config", {})
    try:
        validated_config = RSSSourceParamContract(**rss_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid RSS source configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    enable_monitor = validated_config.enable_monitor

    rss_source_definition = RSSSourceInterface.get_definition("rss_source", {"rss_source": validated_config.dict()})

    web_scraper_definition = WebScraperInterface.get_definition("web_scraper", {
        "web_scraper_config": validated_config.web_scraper_config,
    })

    transform_config = {
        "schema_transform_config": {
            "summary": {"dtype": "str", "op_type": "select"},
            "title": {"dtype": "str", "op_type": "select"},
            "content": {"from": "page_content", "dtype": "str", "op_type": "rename"},
            "source": {"from": "link", "dtype": "str", "op_type": "rename"}
        }
    }
    schema_transform_definition = SchemaTransformInterface.get_definition("schema_transform", transform_config)

    deserialize_definition = DeserializeInterface.get_definition("deserialize",
                                                                 {"batch_size": validated_config.output_batch_size})

    monitor_m1 = Monitor.get_definition("monitor_m1", {"description": "RSSSourcePipe RSS Source",
                                                       "silence_monitors": not enable_monitor})
    monitor_0 = Monitor.get_definition("monitor_0", {"description": "RSSSourcePipe Web Scraper",
                                                     "silence_monitors": not enable_monitor})
    monitor_1 = Monitor.get_definition("monitor_1", {"description": "RSSSourcePipe Transform",
                                                     "silence_monitors": not enable_monitor})
    monitor_2 = Monitor.get_definition("monitor_2", {"description": "RSSSourcePipe Deserialize",
                                                     "silence_monitors": not enable_monitor})

    # Load modules
    rss_source_module = rss_source_definition.load(builder=builder)
    monitor_m1 = monitor_m1.load(builder=builder)
    web_scraper_module = web_scraper_definition.load(builder=builder)
    monitor_0_module = monitor_0.load(builder=builder)
    transform_module = schema_transform_definition.load(builder=builder)
    monitor_1_module = monitor_1.load(builder=builder)
    deserialize_module = deserialize_definition.load(builder=builder)
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


RSSSourcePipe = ModuleInterface("rss_source_pipe", "morpheus_examples_llm",
                                RSSSourceParamContract)
