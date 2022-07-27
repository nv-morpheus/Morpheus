import dataclasses
import typing

import click

from morpheus.config import PipelineModes


@dataclasses.dataclass
class StageInfo:
    name: str
    build: typing.Callable[[], click.Command]
    modes: typing.List[PipelineModes] = dataclasses.field(default_factory=list)

    def supports_mode(self, mode: PipelineModes):
        if (mode is None):
            return True

        if (self.modes is None or len(self.modes) == 0):
            return True

        return mode in self.modes


class StageRegistry:

    _registered_stages: typing.Dict[str, StageInfo] = {}

    @staticmethod
    def add_stage_info(stage: StageInfo):

        StageRegistry._registered_stages[stage.name] = stage

    @staticmethod
    def get_stage_info(stage_name: str, mode: PipelineModes = None, raise_missing=False) -> StageInfo:

        if (stage_name not in StageRegistry._registered_stages):
            if (raise_missing):
                raise RuntimeError("Could not find stage '{}' in registry".format(stage_name))
            else:
                return None

        stage_info = StageRegistry._registered_stages[stage_name]

        # Now check the modes
        if (stage_info.supports_mode(mode)):
            return stage_info

        # Found but no match on mode
        if (raise_missing):
            raise RuntimeError("Found stage '{}' in registry, but it does not support pipeline mode: {}".format(
                stage_name, mode))
        else:
            return None

    @staticmethod
    def get_registered_names(mode: PipelineModes = None) -> typing.List[str]:

        # Loop over all registered stages and validate the mode
        stage_names: typing.List[str] = [
            name for name, stage_info in StageRegistry._registered_stages.items() if stage_info.supports_mode(mode)
        ]

        return stage_names
