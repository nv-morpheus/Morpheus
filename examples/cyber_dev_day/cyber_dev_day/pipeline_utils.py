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

from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.utils.langchain_llm_client_wrapper import LangchainLLMClientWrapper
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler

from .checklist_node import CVEChecklistNode
from .config import EngineAgentConfig
from .config import EngineConfig
from .tools import SBOMChecker

logger = logging.getLogger(__name__)


def build_agent_executor(config: EngineAgentConfig) -> AgentExecutor:

    llm_service = LLMService.create(config.model.service.type, **config.model.service.model_dump(exclude={"type"}))

    llm_client = llm_service.get_client(**config.model.model_dump(exclude={"service"}))

    # Wrap the Morpheus client in a LangChain compatible wrapper
    langchain_llm = LangchainLLMClientWrapper(client=llm_client)

    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools: list[Tool] = []

    if (config.sbom.data_file is not None):

        # Load the SBOM
        sbom_checker = SBOMChecker.from_csv(config.sbom.data_file)

        tools.append(
            Tool(name="SBOM Package Checker",
                 func=sbom_checker.sbom_checker,
                 description=("useful for when you need to check the Docker container's software bill of "
                              "materials (SBOM) to get whether or not a given library is in the container. "
                              "Input should be the name of the library or software. If the package is "
                              "present a version number is returned, otherwise False is returned if the "
                              "package is not present.")))

    if (config.code_repo.faiss_dir is not None):
        embeddings = HuggingFaceEmbeddings(model_name=config.code_repo.embedding_model_name,
                                           model_kwargs={'device': 'cuda'},
                                           encode_kwargs={'normalize_embeddings': False})

        # load code vector DB
        code_vector_db = FAISS.load_local(folder_path=config.code_repo.faiss_dir,
                                          embeddings=embeddings,
                                          allow_dangerous_deserialization=True)
        code_qa_tool = RetrievalQA.from_chain_type(llm=langchain_llm,
                                                   chain_type="stuff",
                                                   retriever=code_vector_db.as_retriever())
        tools.append(
            Tool(name="Docker Container Code QA System",
                 func=code_qa_tool.run,
                 description=("useful for when you need to check if an application or any dependency within "
                              "the Docker container uses a function or a component of a library.")))

    agent_executor = initialize_agent(tools, langchain_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def build_cve_llm_engine(config: EngineConfig) -> LLMEngine:

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("checklist", inputs=["/extracter"], node=CVEChecklistNode(config=config.checklist))

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=build_agent_executor(config=config.agent)))

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine
