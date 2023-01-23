# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import click

from morpheus.cli import hookimpl
from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes


class DefaultCommandHooks:
    """
    Hooks for registering and collecting stages to execute as part of a morhpeus pipeline
    """

    @hookimpl
    def morpheus_cli_collect_stage_names(self, mode: PipelineModes) -> typing.List[str]:
        """
        Loop over the existing stage registry and return the names
        """
        command_names = StageRegistry.get_registered_names(mode=mode)

        return command_names

    @hookimpl
    def morpheus_cli_make_stage_command(self, mode: PipelineModes, stage_name: str) -> click.Command:
        """
        Add a stage command to a pipeline mode based on info in the stage registry
        """

        stage_info = StageRegistry.get_stage_info(stage_name=stage_name, mode=mode, raise_missing=False)

        if (stage_info is None):
            return None

        command = stage_info.build()

        return command
