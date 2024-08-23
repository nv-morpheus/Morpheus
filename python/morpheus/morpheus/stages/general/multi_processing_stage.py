from abc import abstractmethod
import typing

from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.shared_process_pool import SharedProcessPool
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
import mrc
import mrc.core.operators as ops

InputT = typing.TypeVar('InputT')
OutputT = typing.TypeVar('OutputT')


class MultiProcessingBaseStage(SinglePortStage, typing.Generic[InputT, OutputT]):

    def __init__(self, *, c: Config, process_pool_usage: float):
        super().__init__(c=c)

        self._process_pool_usage = process_pool_usage

    @property
    def name(self) -> str:
        return "multi-processing-base-stage"

    def accepted_types(self) -> typing.Tuple:
        return (InputT, )

    def compute_schema(self, schema: StageSchema):
        return super().compute_schema(schema)

    @abstractmethod
    def _on_data(self, data: InputT) -> OutputT:
        pass


class MultiProcessingStage(MultiProcessingBaseStage[InputT, OutputT]):

    def __init__(self,
                 *,
                 c: Config,
                 process_pool_usage: float,
                 process_fn: typing.Callable[[InputT], OutputT],
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage)

        self._process_fn = process_fn
        self._shared_process_pool = SharedProcessPool()
        self._shared_process_pool.set_usage(self.name, self._process_pool_usage)
        if max_in_flight_messages is None:
            # set the multiplier to 1.5 to keep the workers busy
            self._max_in_flight_messages = self._shared_process_pool.total_max_workers * 1.5
        else:
            self._max_in_flight_messages = max_in_flight_messages

    @property
    def name(self) -> str:
        return "multi-processing-stage"

    def _on_data(self, data: InputT) -> OutputT:

        future = self._shared_process_pool.submit_task(self.name, self._process_fn, data)
        result = future.result()

        return result

    @staticmethod
    def create(*, c: Config, process_fn: typing.Callable[[InputT], OutputT], process_pool_usage: float):

        return MultiProcessingStage[InputT, OutputT](c=c, process_pool_usage=process_pool_usage, process_fn=process_fn)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.name, ops.map(self._on_data))
        node.lanuch_options.pe_count = self._max_in_flight_messages

        builder.make_edge(input_node, node)

        return node


# pipe = LinearPipeline(config)

# # ...add other stages...

# # You can derive from the base class if you need to use self inside the process function
# class MyCustomMultiProcessStage(MultiProcessStage[ControlMessage, ControlMessage]):

#     def __init__(self, *, c: Config, process_pool_usage: float, add_column_name: str):
#         super().__init__(self, c=c, process_pool_usage=process_pool_usage)

#         self._add_column_name = add_column_name

#     def _on_data(self, data: ControlMessage) -> ControlMessage:

#         with data.payload().mutable_dataframe() as df:
# 		df[self._add_column_name] = "hello"

#         return data

# # Add an instance of the custom stage
# pipe.add_stage(MyCustomMultiProcessStage(c=config, process_pool_usage, add_column_name="NewCol")

# # If you just want to supply a function pointer
# def print_process_id(message):
#     print(os.pid())
#     return message

# pipe.add_stage(MultiProcessingStage.create(c=config, process_fn=print_process_id))
