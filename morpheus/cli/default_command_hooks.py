import typing

import click

from morpheus.cli import hookimpl
from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes


class DefaultCommandHooks:

    @hookimpl
    def morpheus_cli_collect_stage_names(self, mode: PipelineModes) -> typing.List[str]:

        # Loop over the existing stage registry and return the names
        command_names = StageRegistry.get_registered_names(mode=mode)

        return command_names

    @hookimpl
    def morpheus_cli_make_stage_command(self, mode: PipelineModes, stage_name: str) -> click.Command:

        stage_info = StageRegistry.get_stage_info(stage_name=stage_name, mode=mode, raise_missing=False)

        if (stage_info is None):
            return None

        command = stage_info.build()

        return command
