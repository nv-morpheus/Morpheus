<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

**General architectural ideas**

We build three libraries:
- libmorpheus : defines all the python aware library code for morpheus, and interface proxies for python modules.
  - Interface proxies are designed to provide a single consolidated point of interaction between the morpheus 
  library code and their associated pybind11 module definitions.
  - Please avoid declaring adhoc functions/interfaces that link to python modules.
- libmorpheus_utils : matx and table manipulation functions.
- libcudf_helpers : small bridge module used to extract cython based dataframe, and series information from cuDF.


Python modules should be defined in `_lib/src/python_modules`, with an associated cmake declaration in 
`_lib/cmake/<module_name>.cmake` which can be included in `_lib/CMakeLists.txt`.