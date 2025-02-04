# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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


# The purpose of this function is to allow loading the current directory as a module. This allows relative imports and
# more specifically `..common` to function correctly
def run_cli():
    import os
    import sys

    examples_dir = os.path.dirname(os.path.dirname(__file__))

    if (examples_dir not in sys.path):
        sys.path.append(examples_dir)

    from llm.cli import cli

    cli(obj={}, auto_envvar_prefix='MORPHEUS_LLM', show_default=True)


if __name__ == '__main__':
    run_cli()
