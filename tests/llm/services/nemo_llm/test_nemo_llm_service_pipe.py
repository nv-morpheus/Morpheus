# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cudf

from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.llm.llm_engine_stage import LLMEngineStage
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.services.nemo_llm_service import NeMoLLMService
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes


def _build_engine(model_name: str):
    llm_service = NeMoLLMService()
    llm_clinet = llm_service.get_client(model_name=model_name)

    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())
    engine.add_node("completion", inputs=["/extracter"], node=LLMGenerateNode(llm_client=llm_clinet))
    engine.add_task_handler(inputs=["/completion"], handler=SimpleTaskHandler())

    return engine


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.usefixtures("ngc_api_key")
@pytest.mark.parametrize("model_name", ["gpt-43b-002"])
def test_completion_pipe(config: Config, model_name: str):
    """
    Loosely patterned after `examples/llm/completion`
    """

    source_df = cudf.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "What is the capital of Spain?",
            "What is the capital of Italy?",
            "What is the capital of Germany?",
            "What is the capital of United Kingdom?",
            "What is the capital of China?",
            "What is the capital of Japan?",
            "What is the capital of India?",
            "What is the capital of Brazil?",
            "What is the capital of United States?",
        ]
    })

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["prompt"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    messages = sink.get_messages()
    result_df = concat_dataframes(messages)

    # We don't want to check for specific responses from Nemo, we just want to ensure we received non-empty responses
    assert (result_df['response'].str.strip() != '').all()
