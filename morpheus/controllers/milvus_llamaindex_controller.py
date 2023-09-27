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

import typing

import pandas as pd
from llama_index import Document
from llama_index import StorageContext
from llama_index import VectorStoreIndex
from llama_index.vector_stores import MilvusVectorStore


class MilvusLlamaIndexController:
    """
    """

    def __init__(self, host: str, port: str, collection_name: str, **kwargs: typing.Any):
        self.milvus_vector_store = MilvusVectorStore(host=host, port=port, collection_name=collection_name, **kwargs)

    def transform(self, df: pd.DataFrame, document_column: str, **kwargs: typing.Any):
        """
        """
        documents = []
        for idx, row in df.iterrows():
            document_text = row[document_column]
            document = Document(text=document_text, node_id=str(idx))
            documents.append(document)
        return documents

    def store(self, documents: list):
        """
        """
        storage_context = StorageContext.from_defaults(vector_store=self.milvus_vector_store)
        vector_store = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return vector_store

    def query(self, query: str, vector_store: typing.Any, **kwargs: typing.Any):
        """
        """
        query_engine = vector_store.as_query_engine()
        response = query_engine.query(query, **kwargs)
        return response
