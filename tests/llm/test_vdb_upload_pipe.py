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
"""Mimic the examples/llm/vdb_upload/pipeline.py example"""

import json
import os
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from _utils import TEST_DIRS
from _utils import mk_async_infer
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

EMBEDDING_SIZE = 384
MODEL_MAX_BATCH_SIZE = 64
MODEL_FEA_LENGTH = 512


def _run_pipeline(config: Config,
                  milvus_server_uri: str,
                  collection_name: str,
                  rss_files: list[str],
                  utils_mod: types.ModuleType,
                  web_scraper_stage_mod: types.ModuleType):

    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.feature_length = MODEL_FEA_LENGTH
    config.edge_buffer_size = 128
    config.class_labels = [str(i) for i in range(EMBEDDING_SIZE)]

    pipe = LinearPipeline(config)

    pipe.set_source(
        RSSSourceStage(config, feed_input=rss_files, batch_size=128, run_indefinitely=False, enable_cache=False))
    pipe.add_stage(web_scraper_stage_mod.WebScraperStage(config, chunk_size=MODEL_FEA_LENGTH, enable_cache=False))
    pipe.add_stage(DeserializeStage(config))

    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt'),
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='page_content'))

    pipe.add_stage(
        TritonInferenceStage(config, model_name='test-model', server_url='test:0000', force_convert_inputs=True))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=collection_name,
                             resource_kwargs=utils_mod.build_milvus_config(embedding_size=EMBEDDING_SIZE),
                             recreate=True,
                             service="milvus",
                             uri=milvus_server_uri))
    pipe.run()


@pytest.mark.milvus
@pytest.mark.use_python
@pytest.mark.use_pandas
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/common/web_scraper_stage.py')
])
@mock.patch('requests.Session')
@mock.patch('tritonclient.grpc.InferenceServerClient')
def test_vdb_upload_pipe(mock_triton_client: mock.MagicMock,
                         mock_requests_session: mock.MagicMock,
                         config: Config,
                         dataset: DatasetManager,
                         milvus_server_uri: str,
                         import_mod: list[types.ModuleType]):
    # We're going to use this DF to both provide values to the mocked Tritonclient,
    # but also to verify the values in the Milvus collection.
    expected_values_df = dataset["service/milvus_rss_data.json"]

    with open(os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_web_responses.json'), encoding='utf-8') as fh:
        web_responses = json.load(fh)

    # Mock Triton results
    mock_metadata = {
        "inputs": [{
            "name": "input_ids", "datatype": "INT32", "shape": [-1, MODEL_FEA_LENGTH]
        }, {
            "name": "attention_mask", "datatype": "INT32", "shape": [-1, MODEL_FEA_LENGTH]
        }],
        "outputs": [{
            "name": "output", "datatype": "FP32", "shape": [-1, EMBEDDING_SIZE]
        }]
    }
    mock_model_config = {"config": {"max_batch_size": 256}}

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    mock_result_values = expected_values_df['embedding'].to_list()
    inf_results = np.split(mock_result_values,
                           range(MODEL_MAX_BATCH_SIZE, len(mock_result_values), MODEL_MAX_BATCH_SIZE))

    # The triton client is going to perform a logits function, calculate the inverse of it here
    inf_results = [np.log((1.0 / x) - 1.0) * -1 for x in inf_results]

    async_infer = mk_async_infer(inf_results)
    mock_triton_client.async_infer.side_effect = async_infer

    # Mock requests, since we are feeding the RSSSourceStage with a local file it won't be using the
    # requests lib, only web_scraper_stage.py will use it.
    def mock_get_fn(url: str):
        mock_response = mock.MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = web_responses[url]
        return mock_response

    mock_requests_session.return_value = mock_requests_session
    mock_requests_session.get.side_effect = mock_get_fn

    (utils_mod, web_scraper_stage_mod) = import_mod
    collection_name = "test_vdb_upload_pipe"
    rss_source_file = os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_rss_feed.xml')

    _run_pipeline(config=config,
                  milvus_server_uri=milvus_server_uri,
                  collection_name=collection_name,
                  rss_files=[rss_source_file],
                  utils_mod=utils_mod,
                  web_scraper_stage_mod=web_scraper_stage_mod)

    milvus_service = MilvusVectorDBService(uri=milvus_server_uri)
    resource_service = milvus_service.load_resource(name=collection_name)

    assert resource_service.count() == len(expected_values_df)

    db_results = resource_service.query("", offset=0, limit=resource_service.count())
    db_df = pd.DataFrame(sorted(db_results, key=lambda k: k['id']))

    # The comparison function performs rounding on the values, but is unable to do so for array columns
    dataset.assert_compare_df(db_df, expected_values_df[db_df.columns], exclude_columns=['id', 'embedding'])
    db_emb = db_df['embedding']
    expected_emb = expected_values_df['embedding']

    for i in range(resource_service.count()):
        db_emb_row = pd.DataFrame(db_emb[i], dtype=np.float32)
        expected_emb_row = pd.DataFrame(expected_emb[i], dtype=np.float32)
        dataset.assert_compare_df(db_emb_row, expected_emb_row)
