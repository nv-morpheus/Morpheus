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
"""
DFP module definitions, each module is automatically registered when imported
"""

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
from dfp.modules import dfp_monitor  # noqa: F401
from dfp.modules import dfp_split_users  # noqa: F401
from dfp.modules import dfp_data_prep  # noqa: F401
from dfp.modules import dfp_inference  # noqa: F401
from dfp.modules import dfp_postprocessing  # noqa: F401
from dfp.modules import dfp_preproc  # noqa: F401
from dfp.modules import dfp_rolling_window  # noqa: F401
from dfp.modules import dfp_training  # noqa: F401
from dfp.modules import dfp_inference_pipe  # noqa: F401
from dfp.modules import dfp_training_pipe  # noqa: F401
from dfp.modules import dfp_deployment  # noqa: F401

__all__ = []
