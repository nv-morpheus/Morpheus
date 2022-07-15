#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# set -x

# Call clang-tidy adding warnings-as-errors option. Currently this is not
# possible with clang-tidy-diff.py until this is merged:
# https://reviews.llvm.org/D49864

# Also add -fno-caret-diagnostics to prevent clangs own compiler warnings from
# coming through:
# https://github.com/llvm/llvm-project/blob/3f3faa36ff3d84af3c3ed84772d7e4278bc44ff1/libc/cmake/modules/LLVMLibCObjectRules.cmake#L226
echo ${CLANG_TIDY:-$(which clang-tidy)}
echo ${CONDA_BUILD_SYSROOT}/usr/include
echo "$@"
echo "---------------"
${CLANG_TIDY:-$(which clang-tidy)} --warnings-as-errors='*' --extra-arg=-fno-caret-diagnostics "$@"
