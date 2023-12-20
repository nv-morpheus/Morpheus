# Copyright (c) 2023, NVIDIA CORPORATION.
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
import os

import pymilvus
from haystack.agents import Agent
from haystack.agents import Tool
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.embeddings import HuggingFaceEmbeddings

from llama_index import VectorStoreIndex
from llama_index.vector_stores import MilvusVectorStore
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaToolkit

from morpheus.llm.services.nemo_llm_service import NeMoLLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus.service.vdb.utils import VectorDBServiceFactory

logger = logging.getLogger(__name__)


def build_huggingface_embeddings(model_name: str, model_kwargs: dict = None, encode_kwargs: dict = None):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embeddings


def build_llm_service(model_name: str, llm_service: str, tokens_to_generate: int, **model_kwargs):
    lowered_llm_service = llm_service.lower()
    if (lowered_llm_service == 'nemollm'):
        model_kwargs['tokens_to_generate'] = tokens_to_generate
        llm_service = NeMoLLMService()
    elif (lowered_llm_service == 'openai'):
        model_kwargs['max_tokens'] = tokens_to_generate
        llm_service = OpenAIChatService()
    else:
        # TODO(Devin) : Add additional options
        raise RuntimeError(f"Unsupported LLM service name: {llm_service}")

    return llm_service.get_client(model_name, **model_kwargs)


def build_milvus_config(embedding_size: int):
    milvus_resource_kwargs = {
        "index_conf": {
            "field_name": "embedding",
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        },
        "schema_conf": {
            "enable_dynamic_field": True,
            "schema_fields": [
                pymilvus.FieldSchema(name="id",
                                     dtype=pymilvus.DataType.INT64,
                                     description="Primary key for the collection",
                                     is_primary=True,
                                     auto_id=True).to_dict(),
                pymilvus.FieldSchema(name="title",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The title of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="link",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The URL of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="summary",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The summary of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="page_content",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="A chunk of text from the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="embedding",
                                     dtype=pymilvus.DataType.FLOAT_VECTOR,
                                     description="Embedding vectors",
                                     dim=embedding_size).to_dict(),
            ],
            "description": "Test collection schema"
        }
    }

    return milvus_resource_kwargs


def build_milvus_service(embedding_size: int, uri: str = "http://localhost:19530"):
    milvus_resource_kwargs = build_milvus_config(embedding_size)

    vdb_service: MilvusVectorDBService = VectorDBServiceFactory.create_instance("milvus",
                                                                                uri=uri,
                                                                                **milvus_resource_kwargs)

    return vdb_service


def build_rss_urls():
    return [
        "https://www.theregister.com/security/headlines.atom",
        "https://isc.sans.edu/dailypodcast.xml",
        "https://threatpost.com/feed/",
        "http://feeds.feedburner.com/TheHackersNews?format=xml",
        "https://www.bleepingcomputer.com/feed/",
        "https://therecord.media/feed/",
        "https://blog.badsectorlabs.com/feeds/all.atom.xml",
        "https://krebsonsecurity.com/feed/",
        "https://www.darkreading.com/rss_simple.asp",
        "https://blog.malwarebytes.com/feed/",
        "https://msrc.microsoft.com/blog/feed",
        "https://securelist.com/feed",
        "https://www.crowdstrike.com/blog/feed/",
        "https://threatconnect.com/blog/rss/",
        "https://news.sophos.com/en-us/feed/",
        "https://www.us-cert.gov/ncas/current-activity.xml",
        "https://www.csoonline.com/feed",
        "https://www.cyberscoop.com/feed",
        "https://research.checkpoint.com/feed",
        "https://feeds.fortinet.com/fortinet/blog/threat-research",
        "https://www.mcafee.com/blogs/rss",
        "https://www.digitalshadows.com/blog-and-research/rss.xml",
        "https://www.nist.gov/news-events/cybersecurity/rss.xml",
        "https://www.sentinelone.com/blog/rss/",
        "https://www.bitdefender.com/blog/api/rss/labs/",
        "https://www.welivesecurity.com/feed/",
        "https://unit42.paloaltonetworks.com/feed/",
        "https://mandiant.com/resources/blog/rss.xml",
        "https://www.wired.com/feed/category/security/latest/rss",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        "https://blog.google/threat-analysis-group/rss/",
        "https://intezer.com/feed/",
    ]


def build_langchain_agent_executor(model_name: str) -> AgentExecutor:
    """
    Builds a LangChain agent executor.

    Parameters
    ----------
    model_name : str
        LLM Model name.

    Returns
    -------
    AgentExecutor
        Langchain AgentExecutor instance.
    """

    llm = OpenAI(model=model_name, temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def build_llama_index_agent_executor(model_name: str,) -> AgentExecutor:

    llm = OpenAI(model=model_name, temperature=0)

    vector_store = MilvusVectorStore(dim=1536, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()

    query_enige_config = IndexToolConfig(query_engine=query_engine, 
                                         name="Brief biography of celebrities",
                                         description="Category of the celebrity, one of [Sports, Entertainment, Business, Music]",
                                         tool_kwargs={"return_direct": False}
                                         )
    toolkit = LlamaToolkit(index_configs=[query_enige_config])
    langchain_tools = load_tools(["llm-math"], llm=llm)
    tools = (toolkit.get_tools() + langchain_tools)
    
    # The 'initialize_agent' function in langchain is being utilized by llamaindex's 'create_llama_agent' behind the scenes.
    agent_executor = initialize_agent(tools,
                                      llm,
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose=True)
    
    return agent_executor

def build_haystack_agent(model_name: str) -> Agent:
    """
    Builds a Haystack agent.

    Parameters
    ----------
    model_name : str
        LLM Model name.

    Returns
    -------
    Agent
        Haystack Agent instance.
    """

    search_key = os.environ.get("SERPERDEV_API_KEY")
    if not search_key:
        raise ValueError("Ensure to configure the SERPERDEV_API_KEY environment variable.")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Ensure to configure the OPENAI_API_KEY environment variable.")

    web_prompt_node = PromptNode(
        model_name,
        api_key=openai_key,
        max_length=256,
        default_prompt_template="deepset/question-answering",
    )

    calc_prompt_node = PromptNode(model_name,
                                  api_key=openai_key,
                                  default_prompt_template="""
        Calculate the result of the following mathematical expression:

        Expression: ({query})
        """)

    web_retriever = WebRetriever(api_key=search_key)
    web_qa_pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=web_prompt_node)

    prompt_template = PromptTemplate("deepset/zero-shot-react")
    prompt_node = PromptNode(model_name,
                             api_key=os.environ.get("OPENAI_API_KEY"),
                             max_length=512,
                             stop_words=["Observation:"])

    web_qa_tool = Tool(
        name="Search",
        pipeline_or_node=web_qa_pipeline,
        description="Useful when you need to search for answers online.",
        output_variable="results",
    )

    calc_tool = Tool(name="Calculator",
                     pipeline_or_node=calc_prompt_node,
                     description="Useful when you need to math calculations.")

    agent = Agent(prompt_node=prompt_node,
                  prompt_template=prompt_template,
                  tools_manager=ToolsManager([web_qa_tool, calc_tool]))

    return agent
