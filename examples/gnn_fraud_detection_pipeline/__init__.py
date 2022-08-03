from morpheus.cli import hookimpl
from morpheus.cli.stage_registry import LazyStageInfo
from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes

from . import stages


@hookimpl
def morpheus_cli_collect_stages(registry: StageRegistry):

    registry.add_stage_info(
        LazyStageInfo("gnn-fraud-classification",
                      __package__ + ".stages.classification_stage.ClassificationStage",
                      modes=[PipelineModes.OTHER]))
