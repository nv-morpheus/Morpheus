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

from morpheus.modules.input.multi_file_source import multi_file_source  # noqa: F401
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module
from .schema_transform import schema_transform  # noqa: F401
from ...common.pdf_extractor_module import pdf_extractor  # noqa: F401

logger = logging.getLogger(__name__)


@register_module("pdf_file_source_pipe", "morpheus_examples_llm")
def pdf_file_source_pipe(builder: mrc.Builder):
    """
    Sets up a pipeline for processing PDF files.

    This function configures a pipeline that reads PDF files, extracts text from them,
    and then transforms the extracted data according to a specified schema.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder to which the pipeline modules will be added.
    """
    # Load the module configuration from the builder
    module_config = builder.get_current_module_config()

    # Configure and load the multi-file source module
    multi_file_config = {
        "module_id": "multi_file_source",
        "module_name": "multi_file_source",
        "namespace": "morpheus",
        "source_config": module_config["pdf_file_source_config"],
    }
    multi_file_module = load_module(config=multi_file_config, builder=builder)

    # Configure and load the PDF extractor module
    pdf_extractor_config = {
        "module_id": "pdf_extractor",
        "module_name": "pdf_extractor",
        "namespace": "morpheus_examples_llm",
    }
    pdf_extractor_module = load_module(config=pdf_extractor_config, builder=builder)

    # Configure and load the schema transformation module
    transform_config = {
        "module_id": "schema_transform",
        "module_name": "schema_transform",
        "namespace": "morpheus_examples_llm",
        "schema_transform_config": {
            "summary": {"dtype": "str", "op_type": "select"},
            "title": {"dtype": "str", "op_type": "select"},
            "content": {"dtype": "str", "op_type": "select"},
            "source": {"dtype": "str", "op_type": "select"}
        }
    }
    transform_module = load_module(config=transform_config, builder=builder)

    # Connect the modules in the pipeline
    builder.make_edge(multi_file_module.output_port("output"), pdf_extractor_module.input_port("input"))
    builder.make_edge(pdf_extractor_module.output_port("output"), transform_module.input_port("input"))

    # Register the final output of the transformation module
    builder.register_module_output("output", transform_module.output_port("output"))
