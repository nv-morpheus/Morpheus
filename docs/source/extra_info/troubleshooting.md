<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Troubleshooting

**Out of Date Build Cache**

The Morpheus build system, by default, stores all build artifacts and cached output in the `${MORPHEUS_ROOT}/.cache` directory. This cache directory is designed to speed up successive builds but occasionally can get out of date and cause unexpected build errors. In this situation, it's best to completely delete the build and cache directories and restart the build:

```bash
# Delete the build and cache folders
rm -rf ${MORPHEUS_ROOT}/.cache
rm -rf ${MORPHEUS_ROOT}/build

# Restart the build
./scripts/compile.sh
```
**Debugging Python Code**

To debug issues in python code, several launch Visual Studio Code launch configurations have been included in the repo. These launch configurations can be found in `${MORPHEUS_ROOT}/morpheus.code-workspace`. To launch the debugging environment, ensure that Visual Studio Code has opened the morpheus workspace file (File->Open Workspace from File...). Once the workspace has been loaded, the launch configurations should be available in the debugging tab.

**Debugging C++ Code**

Similar to the Python launch configurations, several C++ launch configurations can be found in the Visual Studio Code workspace file. However, unlike the Python configuration, it's necessary to ensure morpheus was compiled in Debug mode in order for breakpoints to work correctly. To build Morpheus in Debug mode, use the following:

```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=Debug" ./scripts/compile.sh
```
