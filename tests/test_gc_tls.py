import gc
import os
import typing

import cupy as cp

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.io.deserializers import read_file_to_df

MORPHEUS_ROOT = os.environ['MORPHEUS_ROOT']


class FftStage(SinglePortStage):
    """
    Simple stage that performs FFT calculation
    """

    @property
    def name(self) -> str:
        return "fft_stage"

    def supports_cpp_node(self) -> bool:
        return False

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def on_next(self, x: typing.Any):
        cp.fft.fft(cp.zeros(10))

        return x

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self.on_next))
        builder.make_edge(input_stream[0], node)
        return node, input_stream[1]


def run_pipe2(df):
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[df]))
    pipe.add_stage(FftStage(config))
    pipe.run()


def run_pipe1(df):
    """
    Simple C++ pipeline where the sink holds on to a reference to the message
    """
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(InMemorySinkStage(config))
    pipe.run()


def main():
    # Disable auto-GC to make collection deterministic
    gc.set_debug(gc.DEBUG_STATS)
    gc.disable()

    df = read_file_to_df(os.path.join(MORPHEUS_ROOT, 'tests/tests_data/filter_probs.csv'), df_type='cudf')

    run_pipe1(df)
    run_pipe2(df)


if __name__ == '__main__':
    main()
