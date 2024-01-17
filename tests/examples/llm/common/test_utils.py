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

import types

import pymilvus

# TODO(Devin)
# build_huggingface_embeddings, build_milvus_service, and build_llm_service


def test_build_milvus_config_with_valid_embedding_size(import_utils: types.ModuleType):
    embedding_size = 128
    config = import_utils.build_milvus_config(embedding_size)

    assert 'index_conf' in config
    assert 'schema_conf' in config

    embedding_field_schema = next(
        (field for field in config['schema_conf']['schema_fields'] if field["name"] == 'embedding'), None)
    assert embedding_field_schema is not None
    assert embedding_field_schema['params']['dim'] == embedding_size


def test_build_milvus_config_uses_correct_field_types(import_utils: types.ModuleType):
    embedding_size = 128
    config = import_utils.build_milvus_config(embedding_size)

    for field in config['schema_conf']['schema_fields']:
        assert 'name' in field
        assert 'type' in field
        assert 'description' in field

        if field['name'] == 'embedding':
            assert field['type'] == pymilvus.DataType.FLOAT_VECTOR
        else:
            assert field['type'] in [pymilvus.DataType.INT64, pymilvus.DataType.VARCHAR]


def test_build_milvus_config_index_configuration(import_utils: types.ModuleType):
    embedding_size = 128
    config = import_utils.build_milvus_config(embedding_size)

    index_conf = config['index_conf']
    assert index_conf['field_name'] == 'embedding'
    assert index_conf['metric_type'] == 'L2'
    assert index_conf['index_type'] == 'HNSW'
    assert 'params' in index_conf
    assert index_conf['params']['M'] == 8
    assert index_conf['params']['efConstruction'] == 64
