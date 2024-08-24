from typing import Tuple
import cudf
import pytest
import os

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.multi_processing_stage import MultiProcessingBaseStage
from morpheus.stages.general.multi_processing_stage import MultiProcessingStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def test_constructor(config: Config):
    stage = MultiProcessingStage.create(c=config, process_fn=lambda x: x, process_pool_usage=0.5)
    assert stage.name == "multi-processing-stage"


class DerivedMultiProcessingStage(MultiProcessingBaseStage[ControlMessage, ControlMessage]):

    def __init__(self,
                 *,
                 c: Config,
                 process_pool_usage: float,
                 add_column_name: str,
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._add_column_name = add_column_name

    @property
    def name(self) -> str:
        return "derived-multi-processing-stage"

    def accepted_types(self) -> Tuple:
        return (ControlMessage, )

    def _on_data(self, data: ControlMessage) -> ControlMessage:
        with data.payload().mutable_dataframe() as df:
            df[self._add_column_name] = "Hello"

        return data

@pytest.mark.use_python
def test_stage_pipe(config: Config, dataset_pandas: DatasetManager):

    config.num_threads = os.cpu_count()
    input_df = dataset_pandas["filter_probs.csv"]

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=True, message_type=ControlMessage))
    pipe.add_stage(DerivedMultiProcessingStage(c=config, process_pool_usage=0.5, add_column_name="new_column"))

    pipe.run()


# if __name__ == "__main__":
#     config = Config()
#     dataset_pandas = DatasetManager()
#     # test_constructor(config)
#     test_stage_pipe(config, dataset_pandas)
