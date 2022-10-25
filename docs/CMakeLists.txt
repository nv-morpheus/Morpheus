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
find_package(Sphinx REQUIRED)

set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_MAN NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_FILE_PATTERNS "*.c; *.cc; *.cpp; *.h; *.hpp; *.cu; *.cuh; *.md")
set(DOXYGEN_EXCLUDE_PATTERNS "*/tests/*" "*/include/nvtext/*")
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
set(DOXYGEN_EXTENSION_MAPPING "cu=C++; cuh=C++")
set(DOXYGEN_BRIEF_MEMBER_DESC YES)
set(DOXYGEN_HIDE_UNDOC_MEMBERS NO)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
set(DOXYGEN_HAVE_DOT YES)
set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
set(DOXYGEN_INTERACTIVE_SVG YES)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)

doxygen_add_docs(
    ${PROJECT_NAME}_doxygen_docs
    "morpheus/_lib"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generate html documentation"
)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(${PROJECT_NAME}_docs ALL
                  COMMAND
                  ${SPHINX_EXECUTABLE} -b html
                  -Dbreathe_projects.morpheus=${DOXYGEN_OUTPUT_DIRECTORY}/xml ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating documentation with Sphinx"
                  DEPENDS ${PROJECT_NAME}_doxygen_docs)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
