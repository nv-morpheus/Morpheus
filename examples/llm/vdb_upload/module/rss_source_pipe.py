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

from morpheus.modules.input.rss_source import rss_source  # noqa: F401
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module
from .schema_transform import schema_transform  # noqa: F401
from ...common.web_scraper import web_scraper  # noqa: F401

logger = logging.getLogger(__name__)


@register_module("rss_source_pipe", "morpheus_examples_llm")
def rss_source_pipe(builder: mrc.Builder):
    module_config = builder.get_current_module_config()

    rss_source_config = {
        "module_id": "rss_source",
        "module_name": "rss_source",
        "namespace": "morpheus",
        "rss_config": module_config["rss_config"],
    }
    rss_source_module = load_module(config=rss_source_config, builder=builder)

    web_scraper_config = {
        "module_id": "web_scraper",
        "module_name": "web_scraper",
        "namespace": "morpheus_examples_llm",
        "web_scraper_config": module_config["web_scraper_config"],
    }
    web_scraper_module = load_module(config=web_scraper_config, builder=builder)

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
    transform_module = load_module(config=transform_config, builder=builder)

    builder.make_edge(rss_source_module.output_port("output"), web_scraper_module.input_port("input"))
    builder.make_edge(web_scraper_module.output_port("output"), transform_module.input_port("input"))

    builder.register_module_output("output", transform_module.output_port("output"))
