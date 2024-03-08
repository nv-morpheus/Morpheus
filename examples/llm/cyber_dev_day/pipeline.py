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

import ast
import logging
import re
import time
from textwrap import dedent

import openai
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.vectorstores.faiss import FAISS

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm import LLMLambdaNode
from morpheus.llm import LLMNode
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.llm.services.utils.langchain_llm_client_wrapper import LangchainLLMClientWrapper
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes

from .config import EngineAgentConfig
from .config import EngineChecklistConfig
from .config import EngineCodeRepoConfig
from .config import EngineConfig
from .config import EngineSBOMConfig
from .config import NeMoLLMModelConfig
from .config import NeMoLLMServiceConfig
from .config import NVFoundationLLMModelConfig
from .config import NVFoundationLLMServiceConfig
from .pipeline_utils import build_llm_engine
from .tools import SBOMChecker

logger = logging.getLogger(__name__)


def pipeline(
    num_threads: int,
    pipeline_batch_size,
    model_max_batch_size,
    model_name,
    repeat_count,
) -> float:

    nemo_service_config = NeMoLLMServiceConfig()
    nvfoundation_service_config = NVFoundationLLMServiceConfig()

    engine_config = EngineConfig(
        checklist=EngineChecklistConfig(model=NeMoLLMModelConfig(service=nemo_service_config,
                                                                 model_name="gpt-43b-002"), ),
        agent=EngineAgentConfig(
            model=NVFoundationLLMModelConfig(service=nvfoundation_service_config, model_name="mixtral_8x7b"),
            sbom=EngineSBOMConfig(data_file=""),
            code_repo=EngineCodeRepoConfig(
                faiss_dir="/home/mdemoret/Repos/morpheus/morpheus-dev2/.tmp/Sherlock/NSPECT-V1TL-NPZI_code_faiss",
                embedding_model_name="Xenova/text-embedding-ada-002"),
        ),
    )

    engine_config = EngineConfig.model_validate({
        'checklist': {
            'model': {
                'service': {
                    'type': 'nemo', 'api_key': None, 'org_id': None
                },
                'model_name': 'gpt-43b-002',
                'customization_id': None,
                'temperature': 0.0,
                'tokens_to_generate': 300
            }
        },
        'agent': {
            'model': {
                'service': {
                    'type': 'nvfoundation', 'api_key': None
                }, 'model_name': 'mixtral_8x7b', 'temperature': 0.0
            },
            'sbom': {
                'data_file': ''
            },
            'code_repo': {
                'faiss_dir': '/home/mdemoret/Repos/morpheus/morpheus-dev2/.tmp/Sherlock/NSPECT-V1TL-NPZI_code_faiss',
                'embedding_model_name': 'Xenova/text-embedding-ada-002'
            }
        }
    })

    logger.info("Using Engine Config: %s", engine_config.model_dump_json(indent=2))

    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    source_dfs = [
        cudf.DataFrame({
            "cve_info": [
                "An issue was discovered in the Linux kernel through 6.0.9. drivers/media/dvb-core/dvbdev.c has a use-after-free, related to dvb_register_device dynamically allocating fops."
            ]
        })
    ]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["cve_info"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=build_llm_engine(engine_config)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)

    logger.info("Pipeline complete. Received %s responses:\n%s", len(messages), responses['response'])

    return start_time
