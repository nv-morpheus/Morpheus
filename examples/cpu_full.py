#!/opt/conda/envs/morpheus/bin/python

import logging
import click
import pandas as pd
import time
import asyncio

from typing import Generator
from morpheus.config import Config, CppConfig, PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source, stage
from morpheus.utils.logger import configure_logging


logger = logging.getLogger("morpheus.{__name__}")


@source
def source_generator() -> Generator[MessageMeta, None, None]:
    for i in range(5):
        time.sleep(1)
        yield MessageMeta(df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))


@stage
def simple_stage(message: MessageMeta) -> MessageMeta:
    logger.debug(f"simple_stage:\n\n{message.df.to_string()}")
    return message


@click.command()
@click.option(
    "--num_threads",
    default=1,
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--pipeline_batch_size",
    default=1,
    type=click.IntRange(min=1),
    help="Internal batch size for the pipeline. Can be much larger than the model batch size",
)
def run_pipeline(num_threads, pipeline_batch_size):
    configure_logging(log_level=logging.DEBUG)

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.OTHER
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size

    pipeline = LinearPipeline(config)

    pipeline.set_source(source_generator(config))
    pipeline.add_stage(simple_stage(config))

    pipeline.run()

async def example():
    print("a")
    await asyncio.sleep(10)
    print("b")
    pass

def run_example():
    asyncio.run(example())

if __name__ == "__main__":
    run_pipeline()
    # run_example()