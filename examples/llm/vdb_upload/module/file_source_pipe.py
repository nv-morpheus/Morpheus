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
from morpheus.modules.input.multi_file_source import multi_file_source  # noqa: F401
from morpheus.modules.preprocess.deserialize import deserialize  # noqa: F401
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import load_module
from morpheus.utils.module_utils import register_module

from ...common.content_extractor_module import file_content_extractor  # noqa: F401
from .schema_transform import schema_transform  # noqa: F401

logger = logging.getLogger(__name__)


# TODO(): Improve module documentation
@register_module("file_source_pipe", "morpheus_examples_llm")
def _file_source_pipe(builder: mrc.Builder):
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
    file_source_config = module_config.get("file_source_config", {})
    enable_monitor = file_source_config.get("enable_monitor", False)

    # Configure and load the multi-file source module
    multi_file_config = {
        "module_id": "multi_file_source",
        "module_name": "multi_file_source",
        "namespace": "morpheus",
        "source_config": file_source_config,
    }

    # Configure and load the file content extractor module
    file_content_extractor_config = {
        "module_id": "file_content_extractor",
        "module_name": "file_content_extractor",
        "namespace": "morpheus_examples_llm",
        "batch_size": module_config.get("batch_size", 32),  # Example configuration option
        "num_threads": module_config.get("num_threads", 10),  # Example configuration option
        "converters_meta": module_config["file_source_config"].get("converters_meta", None)
    }

    # Configure and load the schema transformation module
    transform_config = {
        "module_id": "schema_transform",
        "module_name": "schema_transform",
        "namespace": "morpheus_examples_llm",
        "schema_transform_config": {
            "summary": {
                "dtype": "str", "op_type": "select"
            },
            "title": {
                "dtype": "str", "op_type": "select"
            },
            "content": {
                "dtype": "str", "op_type": "select"
            },
            "source": {
                "dtype": "str", "op_type": "select"
            }
        }
    }

    deserialize_config = {
        "module_id": "deserialize",
        "module_name": "deserialize",
        "namespace": "morpheus",
        "batch_size": file_source_config.get("batch_size", 32),  # Example configuration option
    }

    # TODO(Devin): Create CMTagger to set metadata items

    monitor_1 = Monitor.get_definition("monitor_1", {"description": "FileSourcePipe Transform",
                                                     "silence_monitors": not enable_monitor})
    monitor_2 = Monitor.get_definition("monitor_2", {"description": "File Source Deserialize",
                                                     "silence_monitors": not enable_monitor})

    # Load modules
    multi_file_module = load_module(config=multi_file_config, builder=builder)
    file_content_extractor_module = load_module(config=file_content_extractor_config, builder=builder)
    transform_module = load_module(config=transform_config, builder=builder)
    monitor_1_module = monitor_1.load(builder=builder)
    deserialize_module = load_module(config=deserialize_config, builder=builder)
    monitor_2_module = monitor_2.load(builder=builder)

    # Connect the modules in the pipeline
    builder.make_edge(multi_file_module.output_port("output"), file_content_extractor_module.input_port("input"))
    builder.make_edge(file_content_extractor_module.output_port("output"), transform_module.input_port("input"))
    builder.make_edge(transform_module.output_port("output"), monitor_1_module.input_port("input"))
    builder.make_edge(monitor_1_module.output_port("output"), deserialize_module.input_port("input"))
    builder.make_edge(deserialize_module.output_port("output"), monitor_2_module.input_port("input"))

    # Register the final output of the transformation module
    builder.register_module_output("output", monitor_2_module.output_port("output"))


FileSourcePipe = ModuleInterface("file_source_pipe", "morpheus_examples_llm")
