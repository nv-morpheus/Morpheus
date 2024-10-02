# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# llm library and tests are dependent on a number of pypi packages - fixme
pip install --no-input milvus==2.3.5
pip install --no-input pymilvus==2.3.6
pip install --no-input langchain==0.1.16
pip install --no-input langchain-nvidia-ai-endpoints==0.0.11
pip install --no-input faiss-gpu==1.7.*
pip install --no-input google-search-results==2.4
pip install --no-input nemollm==0.3.5

pytest tests/morpheus_llm
