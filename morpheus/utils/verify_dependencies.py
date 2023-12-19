# Copyright (c) 2023, NVIDIA CORPORATION.
#
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


def _verify_deps(deps: tuple[str], error_message: str, namespace: dict):
    """
    There are some dependencies that are only used by a specific stage, and are not installed by default.
    These packages are imported in a guarded try-except block. It is only when the stage is used that these
    dependencies need to be enforced.

    raise ImportError if any of the dependencies are not installed.
    """
    for dep in deps:
        if dep not in namespace:
            raise ImportError(error_message)
