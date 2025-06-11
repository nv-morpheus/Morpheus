# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Mimic the examples/llm/rag/standalone_pipeline.py example"""

import os
import sys
import types
from unittest import mock

import pytest

import cudf

from _utils import TEST_DIRS
from _utils import assert_results
from _utils.llm import mk_mock_openai_response
from _utils.milvus import populate_milvus
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus_llm.llm import LLMEngine
from morpheus_llm.llm.nodes.extracter_node import ExtracterNode
from morpheus_llm.llm.nodes.rag_node import RAGNode
from morpheus_llm.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus_llm.stages.llm.llm_engine_stage import LLMEngineStage

EMBEDDING_SIZE = 384
QUESTION = "What are some new attacks discovered in the cyber security industry?"
PROMPT = """You are a helpful assistant. Given the following background information:\n
{% for c in contexts -%}
Title: {{ c.title }}
Summary: {{ c.summary }}
Text: {{ c.page_content }}
{% endfor %}

Please answer the following question: \n{{ query }}"""
EXPECTED_RESPONSE = "Ransomware, Phishing, Malware, Denial of Service, SQL injection, and Password Attacks"


@pytest.fixture(name="milvus_rss_collection", scope="module")
def fixture_milvus_rss_collection(milvus_server_uri: str):
    collection_name = "milvus_rss_collection"

    llm_common_path = os.path.join(TEST_DIRS.examples_dir, 'llm/common')
    needs_sys_path_append = llm_common_path not in sys.path
    if needs_sys_path_append:
        sys.path.append(llm_common_path)

    import utils

    df_path = os.path.join(TEST_DIRS.tests_data_dir, 'service/milvus_rss_data.json')
    df = read_file_to_df(df_path, df_type="cudf")
    populate_milvus(milvus_server_uri=milvus_server_uri,
                    collection_name=collection_name,
                    resource_kwargs=utils.build_default_milvus_config(embedding_size=EMBEDDING_SIZE),
                    df=df,
                    overwrite=True)
    yield collection_name

    sys.modules.pop('utils', None)
    if needs_sys_path_append:
        sys.path.remove(llm_common_path)


def _build_engine(llm_service_name: str,
                  model_name: str,
                  milvus_server_uri: str,
                  collection_name: str,
                  utils_mod: types.ModuleType):
    engine = LLMEngine()
    engine.add_node("extracter", node=ExtracterNode())

    vector_service = utils_mod.build_milvus_service(embedding_size=EMBEDDING_SIZE, uri=milvus_server_uri)
    embeddings = utils_mod.build_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2",
                                                        model_kwargs={'device': 'cuda'},
                                                        encode_kwargs={'batch_size': 100})

    llm_service = utils_mod.build_llm_service(model_name=model_name,
                                              llm_service=llm_service_name,
                                              temperature=0.5,
                                              tokens_to_generate=200)

    # Async wrapper around embeddings
    async def calc_embeddings(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    engine.add_node("rag",
                    inputs=["/extracter"],
                    node=RAGNode(prompt=PROMPT,
                                 vdb_service=vector_service.load_resource(collection_name),
                                 embedding=calc_embeddings,
                                 llm_client=llm_service))

    engine.add_task_handler(inputs=["/rag"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config,
                  llm_service_name: str,
                  model_name: str,
                  milvus_server_uri: str,
                  collection_name: str,
                  repeat_count: int,
                  utils_mod: types.ModuleType) -> dict:
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 64

    questions = [QUESTION] * repeat_count
    source_df = cudf.DataFrame({"questions": questions})
    expected_df = cudf.DataFrame({"questions": questions, "response": [EXPECTED_RESPONSE] * repeat_count})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}
    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(DeserializeStage(config, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(
        LLMEngineStage(config,
                       engine=_build_engine(llm_service_name=llm_service_name,
                                            model_name=model_name,
                                            milvus_server_uri=milvus_server_uri,
                                            collection_name=collection_name,
                                            utils_mod=utils_mod)))
    sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))

    pipe.run()

    return sink.get_results()


@pytest.mark.usefixtures("nemollm")
@pytest.mark.milvus
@pytest.mark.use_cudf
@pytest.mark.parametrize("repeat_count", [5])
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'))
def test_rag_standalone_pipe_nemo(config: Config,
                                  mock_nemollm: mock.MagicMock,
                                  milvus_server_uri: str,
                                  repeat_count: int,
                                  milvus_rss_collection: str,
                                  import_mod: types.ModuleType):
    mock_nemollm.post_process_generate_response.side_effect = [{"text": EXPECTED_RESPONSE} for _ in range(repeat_count)]
    results = _run_pipeline(
        config=config,
        llm_service_name="nemollm",
        model_name="test_model",
        milvus_server_uri=milvus_server_uri,
        collection_name=milvus_rss_collection,
        repeat_count=repeat_count,
        utils_mod=import_mod,
    )
    assert_results(results)


@pytest.mark.usefixtures("openai", "restore_environ")
@pytest.mark.milvus
@pytest.mark.use_cudf
@pytest.mark.parametrize("repeat_count", [5])
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'))
def test_rag_standalone_pipe_openai(config: Config,
                                    mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                                    milvus_server_uri: str,
                                    repeat_count: int,
                                    milvus_rss_collection: str,
                                    import_mod: types.ModuleType):
    os.environ['OPENAI_API_KEY'] = "test"

    (mock_client, mock_async_client) = mock_chat_completion
    mock_async_client.chat.completions.create.side_effect = [
        mk_mock_openai_response([EXPECTED_RESPONSE]) for _ in range(repeat_count)
    ]

    results = _run_pipeline(
        config=config,
        llm_service_name="openai",
        model_name="test_model",
        milvus_server_uri=milvus_server_uri,
        collection_name=milvus_rss_collection,
        repeat_count=repeat_count,
        utils_mod=import_mod,
    )
    assert_results(results)
    mock_client.chat.completions.create.assert_not_called()
    mock_async_client.chat.completions.create.assert_called()


@pytest.mark.usefixtures("nemollm")
@pytest.mark.usefixtures("ngc_api_key")
@pytest.mark.milvus
@pytest.mark.use_cudf
@pytest.mark.parametrize("repeat_count", [5])
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'))
def test_rag_standalone_pipe_integration_nemo(config: Config,
                                              milvus_server_uri: str,
                                              repeat_count: int,
                                              milvus_rss_collection: str,
                                              import_mod: types.ModuleType):

    results = _run_pipeline(
        config=config,
        llm_service_name="nemollm",
        model_name="gpt-43b-002",
        milvus_server_uri=milvus_server_uri,
        collection_name=milvus_rss_collection,
        repeat_count=repeat_count,
        utils_mod=import_mod,
    )

    assert results['diff_cols'] == 0
    assert results['total_rows'] == repeat_count
    assert results['matching_rows'] + results['diff_rows'] == repeat_count


@pytest.mark.usefixtures("openai")
@pytest.mark.usefixtures("openai_api_key")
@pytest.mark.milvus
@pytest.mark.use_cudf
@pytest.mark.parametrize("repeat_count", [5])
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'))
def test_rag_standalone_pipe_integration_openai(config: Config,
                                                milvus_server_uri: str,
                                                repeat_count: int,
                                                milvus_rss_collection: str,
                                                import_mod: types.ModuleType):

    results = _run_pipeline(
        config=config,
        llm_service_name="openai",
        model_name="gpt-3.5-turbo",
        milvus_server_uri=milvus_server_uri,
        collection_name=milvus_rss_collection,
        repeat_count=repeat_count,
        utils_mod=import_mod,
    )

    assert results['diff_cols'] == 0
    assert results['total_rows'] == repeat_count
    assert results['matching_rows'] + results['diff_rows'] == repeat_count
