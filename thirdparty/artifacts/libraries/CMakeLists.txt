# SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set default library versions
set(MORPHEUS_TP_LIBMD_VERSION "1.0.4" CACHE STRING "Version of libmd to build")
set(MORPHEUS_TP_LIBBSD_VERSION "0.11.7" CACHE STRING "Version of libbsd to build")
set(MORPHEUS_TP_GDRCOPY_VERSION "2.3" CACHE STRING "Version of gdrcopy to build")

if (MORPHEUS_TP_BUILD_LIBMD)
  add_subdirectory(md)
endif()

if (MORPHEUS_TP_BUILD_LIBBSD)
  add_subdirectory(bsd)
endif()

if (MORPHEUS_TP_BUILD_GDRCOPY)
  add_subdirectory(gdrcopy)
endif()