# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "docs")

find_package(Doxygen REQUIRED dot doxygen)

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_FILE_PATTERNS "*.c; *.cc; *.cpp; *.h; *.hpp; *.cu; *.cuh; *.md")
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
set(DOXYGEN_EXTENSION_MAPPING "cu=C++; cuh=C++")
set(DOXYGEN_BRIEF_MEMBER_DESC YES)
set(DOXYGEN_HIDE_UNDOC_MEMBERS NO)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
set(DOXYGEN_HAVE_DOT YES)
set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
set(DOXYGEN_INTERACTIVE_SVG YES)

doxygen_add_docs(
    ${PROJECT_NAME}_docs
    "morpheus/_lib"
    "LICENSE"
    "README.md"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generate html documentation"
)


# Add the binary dir before including quickstart
list(PREPEND CMAKE_PREFIX_PATH "${PROJECT_BINARY_DIR}")

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
