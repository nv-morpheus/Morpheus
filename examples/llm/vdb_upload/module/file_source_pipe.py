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

from morpheus.modules.general.monitor import Monitor
from morpheus.modules.input.multi_file_source import MultiFileSourceInterface
from morpheus.modules.preprocess.deserialize import DeserializeInterface
from morpheus.utils.module_utils import ModuleInterface
from morpheus.utils.module_utils import register_module
from .schema_transform import SchemaTransformInterface
from ...common.content_extractor_module import FileContentExtractorInterface

logger = logging.getLogger(__name__)


class FileSourceParamContract(BaseModel):
    batch_size: int = 1024
    chunk_overlap: int = 51
    chunk_size: int = 512
    converters_meta: Optional[Dict[Any, Any]] = {}  # Flexible dictionary for converters metadata
    enable_monitor: bool = False
    extractor_config: Optional[Dict[Any, Any]] = {}  # Flexible dictionary for extractor configuration
    filenames: List[str] = Field(default_factory=list)  # List of file paths
    num_threads: int = 1  # Number of threads for processing
    watch: bool = False  # Flag to watch file changes


@register_module("file_source_pipe", "morpheus_examples_llm")
def _file_source_pipe(builder: mrc.Builder):
    """
    Sets up a pipeline for processing file sources.

    This function configures a pipeline that reads files, processes their content
    based on specified configurations, and outputs the processed data. It integrates modules for
    multi-file sourcing, file content extraction, and schema transformation, along with monitoring
    at various stages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder to which the pipeline modules will be added.

    Notes
    -----
    The module configuration can include the following parameters:

    - **file_source_config**: Configuration for the file source module.
      - **batch_size**: Number of files to process in each batch.
      - **chunk_overlap**: Overlap size for chunks in file processing.
      - **chunk_size**: Size of chunks for file processing.
      - **converters_meta**: Metadata for file format converters.
        - **csv**: Configuration for CSV files.
          - **chunk_size**: Chunk size for CSV processing.
          - **text_column_name**: Name of the text column in CSV files.
      - **enable_monitor**: Boolean to enable monitoring for this module.
      - **extractor_config**: Configuration for the file content extractor module.
        - **chunk_size**: Size of chunks for the extractor.
        - **num_threads**: Number of threads for file content extraction.
      - **filenames**: List of file paths to be processed.
      - **watch**: Boolean to watch for file changes.

    The pipeline connects these modules in the following order:
    Multi-File Source -> File Content Extractor -> Schema Transform -> Deserialize,
    with monitoring at each stage.
    """

    module_config = builder.get_current_module_config()
    file_source_config = module_config.get("file_source_config", {})
    try:
        validated_config = FileSourceParamContract(**file_source_config)
    except ValidationError as e:
        error_messages = '; '.join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid file source configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    # Use the validated configuration
    enable_monitor = validated_config.enable_monitor

    # Configure and load the multi-file source module
    multi_file_definition = MultiFileSourceInterface.get_definition("multi_file_source",
                                                                    {"source_config": validated_config.dict()})

    # Configure and load the file content extractor module
    file_content_extractor_config = {
        "batch_size": validated_config.batch_size,
        "num_threads": validated_config.num_threads,
        "converters_meta": validated_config.converters_meta
    }
    extractor_definition = FileContentExtractorInterface.get_definition("file_content_extractor",
                                                                        file_content_extractor_config)

    # Configure and load the schema transformation module
    transform_config = {
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
    schema_transform_definition = SchemaTransformInterface.get_definition("schema_transform", transform_config)

    deserialize_definition = DeserializeInterface.get_definition("deserialize",
                                                                 {"batch_size": validated_config.batch_size})

    monitor_1 = Monitor.get_definition("monitor_1", {"description": "FileSourcePipe Transform",
                                                     "silence_monitors": not enable_monitor})
    monitor_2 = Monitor.get_definition("monitor_2", {"description": "File Source Deserialize",
                                                     "silence_monitors": not enable_monitor})

    # Load modules
    multi_file_module = multi_file_definition.load(builder=builder)
    file_content_extractor_module = extractor_definition.load(builder=builder)
    transform_module = schema_transform_definition.load(builder=builder)
    monitor_1_module = monitor_1.load(builder=builder)
    deserialize_module = deserialize_definition.load(builder=builder)
    monitor_2_module = monitor_2.load(builder=builder)

    # Connect the modules in the pipeline
    builder.make_edge(multi_file_module.output_port("output"), file_content_extractor_module.input_port("input"))
    builder.make_edge(file_content_extractor_module.output_port("output"), transform_module.input_port("input"))
    builder.make_edge(transform_module.output_port("output"), monitor_1_module.input_port("input"))
    builder.make_edge(monitor_1_module.output_port("output"), deserialize_module.input_port("input"))
    builder.make_edge(deserialize_module.output_port("output"), monitor_2_module.input_port("input"))

    # Register the final output of the transformation module
    builder.register_module_output("output", monitor_2_module.output_port("output"))


FileSourcePipe = ModuleInterface("file_source_pipe", "morpheus_examples_llm",
                                 FileSourceParamContract)
FileSourcePipe.print_schema()
