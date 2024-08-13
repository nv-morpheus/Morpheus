# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
"""Base class for HTTP sources."""

import typing

from morpheus.config import Config
from morpheus.io.utils import get_json_reader
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.type_aliases import DataFrameType


class HttpSourceStageBase(PreallocatorMixin, SingleOutputSource):

    def _get_default_payload_to_df_fn(self, config: Config) -> typing.Callable[[str, bool], DataFrameType]:
        reader = get_json_reader(config)

        return lambda payload, lines: reader(payload, lines=lines)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)
