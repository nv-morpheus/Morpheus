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
"""Mixin used by both SinglePortStage & SingleOutputSource."""

from abc import ABC
from abc import abstractmethod


class SingleOutputMixin(ABC):
    """
    Mixin used by both SinglePortStage & SingleOutputSource. Adds a `output_type` abstract method. Implements the
    `output_types` method which invokes `output_type`.
    """

    def output_types(self, parent_output_types: list[type]) -> list[type]:
        """
        Return the output types for this stage. Derived classes should override this method.

        Returns
        -------
        list
            Output types.

        """
        return [self.output_type(parent_output_types)]

    @abstractmethod
    def output_type(self, parent_output_types: list[type]) -> type:
        """
        Return the output type for this stage. Derived classes should override this method.

        Returns
        -------
        type
            Output type.

        """
        pass