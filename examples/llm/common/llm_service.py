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

import logging
from abc import ABC
from abc import abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_batch(self, prompt: list[str]) -> list[str]:
        pass

    @abstractmethod
    async def generate_batch_async(self, prompt: list[str]) -> list[str]:
        pass


class LLMService(ABC):

    @abstractmethod
    def get_client(self, model_name: str, **model_kwargs: dict) -> LLMClient:
        pass
