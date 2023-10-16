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

# import logging

# import click
# from vdb_upload import run as vdb_upload_run

# from morpheus.cli.utils import get_log_levels
# from morpheus.cli.utils import parse_log_level
# from morpheus.utils.logger import configure_logging

# @click.group(name="morpheus_llm", chain=False, invoke_without_command=True, no_args_is_help=True)
# @click.option("--log_level",
#               default=logging.getLevelName(logging.INFO),
#               type=click.Choice(get_log_levels(), case_sensitive=False),
#               callback=parse_log_level,
#               help="Specify the logging level to use.")
# @click.version_option()
# def cli(log_level: int):
#     """Main entrypoint for the LLM Examples"""

#     # Configure the logging
#     configure_logging(log_level=log_level)

#     morpheus_logger = logging.getLogger("morpheus")

#     logger = logging.getLogger(__name__)

#     # Set the parent logger for all of the llm examples to use morpheus so we can take advantage of configure_logging
#     logger.parent = morpheus_logger

# cli.add_command(vdb_upload_run, name='vdb_upload')

def run_cli():
    import os
    import sys

    examples_dir = os.path.dirname(os.path.dirname(__file__))

    if (examples_dir not in sys.path):
        sys.path.append(examples_dir)

    from llm.cli import cli

    cli(obj={}, auto_envvar_prefix='MORPHEUS_LLM', show_default=True, prog_name="morpheus_llm")


if __name__ == '__main__':
    run_cli()
