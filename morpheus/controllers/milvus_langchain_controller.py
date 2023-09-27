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
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.huggingface import OpenAIEmbeddings
from langchain.embeddings.openai import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus


class MilvusLangChainController:
    """
    """

    def __init__(self, host: str, port: str, **kwargs: typing.Any):
        self._host = host
        self._port = port
        self._kwargs = kwargs
        self._embeddings_map = {"openai": OpenAIEmbeddings, "huggingface": HuggingFaceEmbeddings}

    def transform(self, df: pd.DataFrame, document_column: str, **kwargs: typing.Any):
        """
        """
        loader = DataFrameLoader(data_frame=df, page_content_column=document_column)
        documents = loader.load()

        if "split" in kwargs:
            chunk_size = kwargs.get("chunk_size", 1024)
            chunk_overlap = kwargs.get("chunk_overlap", 0)
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(documents)

        return documents

    def store(self, documents: list, **kwargs: typing.Any):
        """
        """
        embedding = kwargs.get("embedding", "openai")
        embeddings = self._embeddings_map[embedding](**kwargs)
        vector_store = Milvus.from_documents(documents,
                                             embedding=embeddings,
                                             connection_args={
                                                 "host": self._host, "port": self._port
                                             })
        return vector_store

    def query(self, query: str, vector_store: typing.Any, **kwargs: typing.Any):
        """
        """
        response = vector_store.similarity_search(query, **kwargs)
        return response
