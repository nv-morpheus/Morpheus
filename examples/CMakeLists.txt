# =============================================================================
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
list(APPEND CMAKE_MESSAGE_CONTEXT "examples")

# Add the binary dir before including examples so they can find the parent project
list(PREPEND CMAKE_PREFIX_PATH "${PROJECT_BINARY_DIR}")

add_subdirectory(developer_guide)
add_subdirectory(data_loss_prevention)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
