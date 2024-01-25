# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from typing import Dict
from typing import List

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

logger = logging.getLogger(__name__)


class CSVConverterSchema(BaseModel):
    chunk_overlap: int = 102  # Example default value
    chunk_size: int = 1024
    text_column_names: List[str]

    class Config:
        extra = "forbid"


class ContentExtractorSchema(BaseModel):
    batch_size: int = 32
    chunk_overlap: int = 51
    chunk_size: int = 512
    converters_meta: Dict[str, Dict] = Field(default_factory=dict)
    num_threads: int = 10

    @validator('converters_meta', pre=True)
    def validate_converters_meta(cls, v):
        validated_meta = {}
        for key, value in v.items():
            if key.lower() == 'csv':
                validated_meta[key] = CSVConverterSchema(**value)
            else:
                validated_meta[key] = value
        return validated_meta

    class Config:
        extra = "forbid"
