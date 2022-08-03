import typing

import click
import pluggy

from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes

hookspec = pluggy.HookspecMarker("morpheus")


@hookspec
def morpheus_cli_collect_stages(registry: StageRegistry):
    pass

@hookspec
def morpheus_cli_collect_stage_names(mode: PipelineModes) -> typing.List[str]:
    pass


@hookspec(firstresult=True)
def morpheus_cli_make_stage_command(mode: PipelineModes, stage_name: str) -> click.Command:
    pass


# @hookspec
# def morpheus_cli_register_command(mode: PipelineModes) -> click.Command:
#     """Have a look at the ingredients and offer your own.

#     :param ingredients: the ingredients, don't touch them!
#     :return: a list of ingredients
#     """
