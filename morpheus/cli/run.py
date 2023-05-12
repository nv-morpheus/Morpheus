# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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


def run_cli():

    from morpheus.cli.commands import cli

    # the `cli` method expects a `ctx` instance which is provided by the `prepare_command` decordator, but pylint
    # is unaware of this and will complain about the missing `ctx` parameter. We can safely ignore this error.
    # pylint: disable=no-value-for-parameter
    cli(obj={}, auto_envvar_prefix='MORPHEUS', show_default=True, prog_name="morpheus")


if __name__ == '__main__':

    run_cli()
