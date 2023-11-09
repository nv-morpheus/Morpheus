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

import os
import types
from unittest import mock

import pytest

import cudf

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

EMBEDDING_SIZE = 384


def _populate_milvus(milvus_server_uri: str, collection_name: str, resource_kwargs: dict, df: cudf.DataFrame):
    milvus_service = MilvusVectorDBService(uri=milvus_server_uri)
    milvus_service.create(collection_name, **resource_kwargs)
    resource_service = milvus_service.load_resource(name=collection_name)
    resource_service.insert_dataframe(name=collection_name, df=df, **resource_kwargs)


def _run_pipeline(config: Config,
                  milvus_server_uri: str,
                  collection_name: str,
                  rss_files: list[str],
                  utils_mod: types.ModuleType,
                  web_scraper_stage_mod: types.ModuleType):
    model_fea_length = 512

    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 64
    config.feature_length = model_fea_length
    config.edge_buffer_size = 128
    config.class_labels = [str(i) for i in range(EMBEDDING_SIZE)]

    pipe = LinearPipeline(config)

    pipe.set_source(RSSSourceStage(config, feed_input=rss_files, batch_size=128, run_indefinitely=False))
    pipe.add_stage(web_scraper_stage_mod.WebScraperStage(config, chunk_size=model_fea_length))
    pipe.add_stage(DeserializeStage(config))

    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt'),
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='page_content'))

    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name='all-MiniLM-L6-v2',
                             server_url='test:0000',
                             force_convert_inputs=True,
                             use_shared_memory=True))

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
@pytest.mark.use_cudf
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'llm/common/utils.py'),
    os.path.join(TEST_DIRS.examples_dir, 'llm/common/web_scraper_stage.py')
])
def test_vdb_upload_pipe(config: Config, milvus_server_uri: str, import_mod: list[types.ModuleType]):
    (utils_mod, web_scraper_stage_mod) = import_mod
    collection_name = "test_vdb_upload_pipe"
    rss_source_file = os.path.join(TEST_DIRS.tests_data_dir, 'service/cisa_rss_feed.xml')
    _run_pipeline(config=config,
                  milvus_server_uri=milvus_server_uri,
                  collection_name=collection_name,
                  rss_files=[rss_source_file],
                  utils_mod=utils_mod,
                  web_scraper_stage_mod=web_scraper_stage_mod)
