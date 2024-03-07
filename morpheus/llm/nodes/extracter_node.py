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

import logging
import typing

from morpheus.llm import LLMContext
from morpheus.llm import LLMNodeBase
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


class ExtracterNode(LLMNodeBase):
    """
    Extracts fields from the DataFrame contained by the message attached to the `LLMContext` and copies them directly
    to the context.

    The list of fields to be extracted is provided by the task's `input_keys` attached to the `LLMContext`.
    """

    def __init__(self, filter_fn: typing.Callable[[DataFrameType], typing.Iterable[bool]] = None):
        super().__init__()
        self._filter_fn = filter_fn

    def get_input_names(self) -> list[str]:
        # This node does not receive its inputs from upstream nodes, but rather from the task itself
        return []

    async def execute(self, context: LLMContext) -> LLMContext:
        # Get the keys from the task
        input_keys: list[str] = typing.cast(list[str], context.task()["input_keys"])

        with context.message().payload().mutable_dataframe() as df:
            if self._filter_fn is not None:
                row_mask = self._filter_fn(df)
                filtered_df = df.loc[row_mask, :]

                # Store the row mask in the context so that the task handler has access to it
                context.set_row_mask(row_mask)
            else:
                filtered_df = df

            input_dicts: list[dict] = filtered_df[input_keys].to_dict(orient="list")

        if (len(input_keys) == 1):
            # Extract just the first key if there is only 1
            context.set_output(input_dicts[input_keys[0]])
        else:
            context.set_output(input_dicts)

        return context
