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

from qdrant_openapi_client.qdrant_client import QdrantClient

from morpheus.controllers.vector_db_controller import VectorDatabaseController


class QdrantVectorDBController(VectorDatabaseController):

    def __init__(self, api_url='http://localhost:6333'):
        self.client = QdrantClient(api_url=api_url)

    def insert(self, name, data, **kwargs):
        pass

    def search(self, name, query=None, **kwargs):
        pass
