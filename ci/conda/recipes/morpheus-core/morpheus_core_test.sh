# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# For now conda tests are disabled on aarch64, largely due to difficulties with installing a cuda enabled version of
# pytorch on aarch64 from a requirements file.
if [[ $(arch) == "aarch64" ]]; then
    exit 0
fi

package_name="morpheus"
file_name="requirements_morpheus_core_arch-$(arch).txt"

# Install requirements if they are included in the package
python3 <<EOF
import subprocess
import sys
import importlib.resources as ir

package_name = "${package_name}"
file_name = "${file_name}"

with ir.as_file(ir.files(anchor=package_name).joinpath(file_name)) as requirements_file:
    subprocess.call(f"pip install -r {requirements_file}".split())
EOF

pytest tests/${package_name}
