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
"""Utilities for testing Morpheus with FAISS"""
from typing import List


class FakeEmbedder:

    def embed_query(self, data: str) -> List[float]:
        # setting data to arbitrary float since constant value will always be returned
        data = 0.0
        return [float(1.0)] * 1023 + [float(0.0) * data]

    def embed_documents(self, data: list) -> List[List[float]]:
        return [[float(3.1)] * 1023 + [float(i)] for i in range(len(data))]

    async def aembed_query(self, data: str) -> List[float]:
        # setting data to arbitrary float since constant value will always be returned
        data = 0.0
        return [float(1.0)] * 1023 + [float(0.0) * data]

    async def aembed_documents(self, data: list) -> List[List[float]]:
        return [[float(3.1)] * 1023 + [float(i)] for i in range(len(data))]
