# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.utils.concat_df import concat_dataframes

from .common import build_common_pipeline

logger = logging.getLogger(__name__)


def pipeline(
    num_threads: int,
    pipeline_batch_size,
    model_max_batch_size,
    model_name,
    repeat_count,
) -> float:
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    source_dfs = [
        cudf.DataFrame(
            {"questions": ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]})
    ]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    sink = build_common_pipeline(config=config, pipe=pipe, task_payload=completion_task, model_name=model_name)

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)

    logger.info("Pipeline complete. Received %s responses:\n%s", len(messages), responses['response'])

    return start_time
