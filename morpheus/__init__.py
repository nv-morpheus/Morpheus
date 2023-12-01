# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
"""Root module for the Morpheus library."""

# ########################### CVE-2023-47248 Mitigation ############################
# Import pyarrow_hotfix as early as possible to ensure that the pyarrow hotfix is applied before any code can use it
# Can be removed after upgrading to pyarrow 14.0.1 or later (which is dictated by cudf)
import pyarrow_hotfix
# ##################################################################################

import logging
import os

# Create a default null logger to prevent log messages from being propagated to users of this library unless otherwise
# configured. Use the `utils.logging` module to configure Morpheus logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import _version  # pylint: disable=wrong-import-position

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
__version__ = _version.get_versions()['version']
