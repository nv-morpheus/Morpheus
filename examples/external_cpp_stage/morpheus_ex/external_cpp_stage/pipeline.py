import click


@click.command()
def run_pipeline():

    import logging
    import os

    from morpheus.config import Config
    from morpheus.pipeline import LinearPipeline
    from morpheus.stages.general.monitor_stage import MonitorStage
    from morpheus.stages.input.file_source_stage import FileSourceStage
    from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
    from morpheus.utils.logger import configure_logging

    from .pass_thru import PassThruStage

    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email_with_addresses.jsonlines')

    config = Config()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    pipeline.add_stage(DeserializeStage(config))

    # Add our own stage
    pipeline.add_stage(PassThruStage(config))

    # Add monitor to record the performance of our new stage
    pipeline.add_stage(MonitorStage(config))

    # Run the pipeline
    pipeline.run()


def cli():
    run_pipeline(obj={}, auto_envvar_prefix='MORPHEUS_EX3', show_default=True, prog_name="morpheus_ex_simple_cpp_stage")


if __name__ == '__main__':
    # Run the CLI by default
    cli()
