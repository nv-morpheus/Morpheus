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
"""Mixins for stages which receicve and emit the same type."""

import types
import typing
from abc import ABC

import typing_utils

from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import is_union_type


class ExplicitPassThruTypeMixin(ABC):
    """
    Mixin intented to be added to stages which accept only a single type, and emit the exact same type.
    This is not intended to be used by stages which declare `typing.Any` as their accepted type.
    """

    ERROR_MSG = "ExplicitPassThruTypeMixin is intended to only be used by stages which accept only a single type."

    def output_type(self, parent_output_types: list[type]) -> type:
        accepted_types = self.accepted_types()
        if len(accepted_types) != 1:
            raise RuntimeError(self.ERROR_MSG)

        if len(set(parent_output_types)) != 1:
            raise RuntimeError(self.ERROR_MSG)

        accepted_type = accepted_types[0]

        # using `typing_utils.issubtype` since `issubclass` does not work with `types.UnionType`
        if (accepted_type is typing.Any or typing_utils.issubtype(accepted_type, types.UnionType)):
            raise RuntimeError(self.ERROR_MSG)

        # Using != rather than `is not` since `int | float == typing.Union[int, float]` is True but
        # `int | float is typing.Union[int, float]` is False.
        if (accepted_type != parent_output_types[0]):
            raise RuntimeError(self.ERROR_MSG)

        return accepted_type


class InferredPassThruTypeMixin(ABC):
    """
    Mixin intented to be added to stages which can accept multiple types or even `typing.Any`, who's output type type
    is inferred from the output types of the parent stages.
    """

    def output_type(self, parent_output_types: list[type]) -> type:

        flattened_types = []
        for parent_type in parent_output_types:
            # TODO: Move this to pipeline.py
            assert parent_type is not typing.Any, "typing.Any is not an acceptable output type"

            if is_union_type(parent_type):
                flattened_types.extend(typing.get_args(parent_type))
            else:
                flattened_types.append(parent_type)

        return greatest_ancestor(*flattened_types)
