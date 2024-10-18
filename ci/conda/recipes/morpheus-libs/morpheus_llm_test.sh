# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install requirements if they are included in the package
python3 <<EOF
import pkgutil
import subprocess
data = pkgutil.get_data("morpheus_llm", "requirements_morpheus_llm.txt")
requirements = data.decode("utf-8")
subprocess.call(f"pip install {requirements}".split())
EOF

pytest tests/morpheus_llm
