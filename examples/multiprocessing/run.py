import mrc
import multiprocessing as mp

from morpheus.messages.multi_message import MultiMessage

from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.config import Config
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage

import cudf

class MyMultiprocessingStage(SinglePortStage):

    @property
    def name(self) -> str:
        return "my-multiprocessing"
    
    def accepted_types(self) -> tuple:
        return (MultiMessage, )
    
    def supports_cpp_node(self):
        return False

    @staticmethod
    def _do_thing(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.subscribe(sub)
    
    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair):
        stream = builder.make_node("my-multiprocessing", mrc.core.operators.build(MyMultiprocessingStage._do_thing))
        builder.make_edge(input_stream[0], stream)
        return stream, input_stream[1]

def run_pipeline():

    config = Config()

    df_input = cudf.DataFrame({ "name": "1", "value": 1 })

    pipeline = LinearPipeline(config)
    pipeline.set_source(InMemorySourceStage(config, [df_input]))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MyMultiprocessingStage(config))
    pipeline.add_stage(SerializeStage(config))
    sink = pipeline.add_stage(InMemorySinkStage(config))

    pipeline.run()

    messages = sink.get_messages()

    print(messages[0].copy_dataframe())

if __name__ == f"__main__":
    run_pipeline()

