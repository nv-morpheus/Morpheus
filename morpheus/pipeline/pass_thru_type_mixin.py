# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
"""Mixin used by stages which are emitting newly constructed DataFrame or MessageMeta instances into the segment."""

import types
import typing
from abc import ABC

ERROR_MSG = "PassThruTypeMixin is intended to only be used by stages which accept only a single type."


class PassThruTypeMixin(ABC):
    """
    Mixin intented to be added to stages which accept only a single type, and emit the exact same type.
    This is not intended to be used by stages which declare `typing.Any` as their accepted type.
    """

    def output_type(self, parent_output_types: list[type]) -> type:
        accepted_types = self.accepted_types()
        if len(accepted_types) != 1:
            raise RuntimeError(ERROR_MSG)

        if len(set(parent_output_types)) != 1:
            raise RuntimeError(ERROR_MSG)

        accepted_type = accepted_types[0]

        if (accepted_type is typing.Any or issubclass(accepted_type, types.UnionType)):
            raise RuntimeError(ERROR_MSG)

        # Using != rather than `is not` since `int | float == typing.Union[int, float]` is True but
        # `int | float is typing.Union[int, float]` is False.
        if (accepted_type != parent_output_types[0]):
            raise RuntimeError(ERROR_MSG)

        return accepted_type
