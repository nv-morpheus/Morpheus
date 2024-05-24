#!/usr/bin/env python
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

import pytest
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from _utils.faiss import FakeEmbedder
from morpheus.service.vdb.faiss_vdb_service import FaissVectorDBResourceService
from morpheus.service.vdb.faiss_vdb_service import FaissVectorDBService

# create FAISS docstore for testing
texts = ["for", "the", "test"]
embeddings = FakeEmbedder()
ids = ["a", "b", "c"]
create_store = FAISS.from_texts(texts, embeddings, ids=ids)
INDEX_NAME = "index"
TMP_DIR_PATH = "/workspace/.tmp/faiss_test_index"
create_store.save_local(TMP_DIR_PATH, INDEX_NAME)


@pytest.fixture(scope="function", name="faiss_service")
def faiss_service_fixture(faiss_test_dir: str, faiss_test_embeddings: list):
    # Fixture for FAISS service; can edit FAISS docstore instantiated outside fixture if need to change
    #  embedding model, et.
    service = FaissVectorDBService(local_dir=faiss_test_dir, embeddings=faiss_test_embeddings)
    yield service


def test_load_resource(faiss_service: FaissVectorDBService):
    resource = faiss_service.load_resource()
    assert isinstance(resource, FaissVectorDBResourceService)
    assert resource._name == "index"


def test_count(faiss_service: FaissVectorDBService):
    docstore = "index"
    count = faiss_service.count(docstore)
    assert count == len(faiss_service._local_dir)


def test_insert(faiss_service: FaissVectorDBService):
    # Test for inserting embeddings (not docs, texts) into docstore
    vector = FakeEmbedder().embed_query(data="hi")
    test_data = list(iter([("hi", vector)]))
    docstore_name = "index"
    response = faiss_service.insert(name=docstore_name, data=test_data)
    assert response == {"status": "success"}


def test_delete(faiss_service: FaissVectorDBService):
    # specify name of docstore and ID to delete
    docstore_name = "index"
    delete_id = "a"
    response_delete = faiss_service.delete(name=docstore_name, expr=delete_id)
    assert response_delete == {"status": "success"}


async def test_similarity_search():
    index_to_id = create_store.index_to_docstore_id
    in_mem_docstore = InMemoryDocstore({
        index_to_id[0]: Document(page_content="for"),
        index_to_id[1]: Document(page_content="the"),
        index_to_id[2]: Document(page_content="test"),
    })

    assert create_store.docstore.__dict__ == in_mem_docstore.__dict__

    query_vec = await embeddings.aembed_query("for")
    output = await create_store.asimilarity_search_by_vector(query_vec, k=1)

    assert output == [Document(page_content="for")]


def test_has_store_object(faiss_service: FaissVectorDBService):
    # create FAISS docstore to test with
    object_store = FAISS.from_texts(texts, embeddings, ids=ids)
    object_name = "store_object_index"
    object_store.save_local(TMP_DIR_PATH, object_name)

    # attempt to load docstore with given index name
    load_attempt = faiss_service.has_store_object(object_name)
    assert load_attempt is True

    # attempt to load docstore with wrong index name
    object_name = "wrong_index_name"
    load_attempt = faiss_service.has_store_object(object_name)
    assert load_attempt is False


def test_create(faiss_service: FaissVectorDBService):
    # Test creating docstore from embeddings
    vector = FakeEmbedder().embed_query(data="hi")
    test_embedding = list(iter([("hi", vector)]))
    docstore_name = "index"
    embeddings_docstore = faiss_service.create(name=docstore_name, text_embeddings=test_embedding)

    # save created docstore
    index_name_embeddings = "embeddings_index"
    embeddings_docstore.save_local(TMP_DIR_PATH, index_name_embeddings)

    # attempt to load created docstore
    load_attempt = faiss_service.has_store_object(index_name_embeddings)

    assert load_attempt is True

    # Test creating docstore from texts
    test_texts = ["for", "the", "test"]
    texts_docstore = faiss_service.create(name=docstore_name, texts=test_texts)

    # save created docstore
    index_name_texts = "texts_index"
    texts_docstore.save_local(TMP_DIR_PATH, index_name_texts)

    # attempt to load created docstore
    load_attempt = faiss_service.has_store_object(index_name_texts)

    assert load_attempt is True

    # Test creating docstore from documents
    test_documents = [Document(page_content="This is for the test.")]
    docs_docstore = faiss_service.create(name=docstore_name, documents=test_documents)

    # save created docstore
    index_name_docs = "docs_index"
    docs_docstore.save_local(TMP_DIR_PATH, index_name_docs)

    # attempt to load created docstore
    load_attempt = faiss_service.has_store_object(index_name_docs)

    assert load_attempt is True
