# Copyright (c) 2024, NVIDIA CORPORATION.
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
"""
Mixins to indicate which execution modes are supported for a given stage. These mixins should be used for any stage
that needs to support execution modes other than the default GPU mode, and the supported execution modes do not change
based upon configuration or runtime conditions.
"""

from abc import ABC

from morpheus.config import ExecutionMode


class CpuOnlyMixin(ABC):
    """
    Mixin intented to be added to stages which support only CPU execution modes.
    """

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        """
        Returns a tuple of supported execution modes of this stage.
        """
        return (ExecutionMode.CPU, )


class GpuAndCpuMixin(ABC):
    """
    Mixin intented to be added to stages which support both GPU and CPU execution modes.
    """

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        """
        Returns a tuple of supported execution modes of this stage.
        """
        return (ExecutionMode.GPU, ExecutionMode.CPU)
