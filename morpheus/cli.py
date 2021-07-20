# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from functools import update_wrapper

import click
import click_completion
from click.globals import get_current_context

from morpheus.config import Config
from morpheus.config import ConfigBase
from morpheus.config import ConfigOnnxToTRT
from morpheus.config import PipelineModes
from morpheus.config import auto_determine_bootstrap
from morpheus.utils.logging import configure_logging

# pylint: disable=line-too-long, import-outside-toplevel, invalid-name, global-at-module-level, unused-argument

# Init click completion to help with installing autocomplete.
click_completion.init()

# NOTE: This file will get executed when hitting <TAB> for autocompletion. Any classes that require a long import time
# should be locally imported. For example, `morpheus.Pipeline` takes a long time to import and must be locally imported
# for each function

DEFAULT_CONFIG = Config.default()

command_kwargs = {
    "context_settings": dict(show_default=True, ),
    "no_args_is_help": True,
}

ALIASES = {
    "pipeline": "pipeline-nlp",
}

global logger
logger = logging.getLogger("morpheus.cli")


class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name]
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)


def _without_empty_args(passed_args):
    return {k: v for k, v in passed_args.items() if v is not None}


def without_empty_args(f):
    """
    Removes keyword arguments that have a None value
    """
    def new_func(*args, **kwargs):
        kwargs = _without_empty_args(kwargs)
        return f(get_current_context(), *args, **kwargs)

    return update_wrapper(new_func, f)


def show_defaults(f):
    """
    Ensures the click.Context has `show_defaults` set to True. (Seems like a bug currently)
    """
    def new_func(*args, **kwargs):
        ctx: click.Context = get_current_context()
        ctx.show_default = True
        return f(*args, **kwargs)

    return update_wrapper(new_func, f)


def _apply_to_config(config: ConfigBase = None, **kwargs):
    config = Config.get() if config is None else config

    for param in kwargs:
        if hasattr(config, param):
            setattr(config, param, kwargs[param])
        else:
            warnings.warn(f"No config option matches for {param}")

    return config


def prepare_command(config: ConfigBase = None):
    def inner_prepare_command(f):
        """Preparse command for use. Combines @without_empty_args, @show_defaults and @click.pass_context

        Args:
            f ([type]): [description]
        """
        def new_func(*args, **kwargs):
            ctx: click.Context = get_current_context()
            ctx.show_default = True

            kwargs = _without_empty_args(kwargs)

            # Apply the config if desired
            if (config):
                _apply_to_config(config, **kwargs)

            return f(ctx, *args, **kwargs)

        return update_wrapper(new_func, f)

    return inner_prepare_command


log_levels = list(logging._nameToLevel.keys())

if ("NOTSET" in log_levels):
    log_levels.remove("NOTSET")


def _parse_log_level(ctx, param, value):
    x = logging._nameToLevel.get(value.upper(), None)
    if x is None:
        raise click.BadParameter('Must be one of {}. Passed: {}'.format(", ".join(logging._nameToLevel.keys()), value))
    return x


@click.group(chain=False, invoke_without_command=True, cls=AliasedGroup, **command_kwargs)
@click.option('--debug/--no-debug', default=False)
@click.option("--log_level",
              default=logging.getLevelName(DEFAULT_CONFIG.log_level),
              type=click.Choice(log_levels, case_sensitive=False),
              callback=_parse_log_level,
              help="Specify the logging level to use.")
@click.option('--log_config_file',
              default=DEFAULT_CONFIG.log_config_file,
              type=click.Path(exists=True, dir_okay=False),
              help=("Config file to use to configure logging. Use only for advanced situations. "
                    "Can accept both JSON and ini style configurations"))
@click.version_option()
@prepare_command(Config.get())
def cli(ctx: click.Context,
        log_level: int = DEFAULT_CONFIG.log_level,
        log_config_file: str = DEFAULT_CONFIG.log_config_file,
        **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    # Configure the logging
    configure_logging(log_level=log_level, log_config_file=log_config_file)

    # Re-get the logger class
    global logger
    logger = logging.getLogger("morpheus.cli")


@cli.group(short_help="Run a utility tool", **command_kwargs)
@prepare_command()
def tools(ctx: click.Context, **kwargs):

    pass


@tools.command(short_help="Converts an ONNX model to a TRT engine", **command_kwargs)
@click.option("--input_model", type=click.Path(exists=True, readable=True), required=True)
@click.option("--output_model", type=click.Path(exists=False, writable=True), required=True)
@click.option('--batches', type=(int, int), required=True, multiple=True)
@click.option('--seq_length', type=int, required=True)
@click.option('--max_workspace_size', type=int, default=16000)
@prepare_command(False)
def onnx_to_trt(ctx: click.Context, **kwargs):

    logger.info("Generating onnx file")

    # Convert batches to a list
    kwargs["batches"] = list(kwargs["batches"])

    c = ConfigOnnxToTRT()

    for param in kwargs:
        if hasattr(c, param):
            setattr(c, param, kwargs[param])

    from morpheus.utils.onnx_to_trt import gen_engine

    gen_engine(c)


@tools.group(short_help="Utility for installing/updating/removing shell completion for Morpheus", **command_kwargs)
def autocomplete(**kwargs):
    pass


@autocomplete.command(short_help="Show the Morpheus shell command completion code")
@click.option('-i', '--case-insensitive/--no-case-insensitive', help="Case insensitive completion")
@click.argument('shell', required=False, type=click_completion.DocumentedChoice(click_completion.core.shells))
def show(shell, case_insensitive):
    """Show the click-completion-command completion code"""
    extra_env = {'_CLICK_COMPLETION_COMMAND_CASE_INSENSITIVE_COMPLETE': 'ON'} if case_insensitive else {}
    click.echo(click_completion.core.get_code(shell, extra_env=extra_env))


@autocomplete.command(short_help="Install the Morpheus shell command completion")
@click.option('--append/--overwrite', help="Append the completion code to the file", default=None)
@click.option('-i', '--case-insensitive/--no-case-insensitive', help="Case insensitive completion")
@click.argument('shell', required=False, type=click_completion.DocumentedChoice(click_completion.core.shells))
@click.argument('path', required=False)
def install(append, case_insensitive, shell, path):
    """Install the click-completion-command completion"""
    extra_env = {'_CLICK_COMPLETION_COMMAND_CASE_INSENSITIVE_COMPLETE': 'ON'} if case_insensitive else {}
    shell, path = click_completion.core.install(shell=shell, path=path, append=append, extra_env=extra_env)
    click.echo('%s completion installed in %s' % (shell, path))


@cli.group(short_help="Run one of the available pipelines", cls=AliasedGroup, **command_kwargs)
@click.option('--num_threads',
              default=os.cpu_count(),
              type=click.IntRange(min=1),
              help="Number of internal pipeline threads to use")
@click.option('--pipeline_batch_size',
              default=DEFAULT_CONFIG.pipeline_batch_size,
              type=click.IntRange(min=1),
              help=("Internal batch size for the pipeline. "
                    "Can be much larger than the model batch size. Also used for Kafka consumers"))
@click.option('--model_max_batch_size',
              default=DEFAULT_CONFIG.model_max_batch_size,
              type=click.IntRange(min=1),
              help="Max batch size to use for the model")
@prepare_command(Config.get())
def run(ctx: click.Context, **kwargs):

    pass


@click.group(short_help=("Place this command before a 'pipeline-*' command "
                         "to run the pipeline with multiple processes using dask"),
             cls=AliasedGroup,
             **command_kwargs)
@prepare_command(Config.get().dask)
def dask(ctx: click.Context, **kwargs):

    click.echo(click.style("Using Dask", fg="yellow"))

    Config.get().use_dask = True


@click.group(chain=True, short_help="Run the inference pipeline with a NLP model", cls=AliasedGroup, **command_kwargs)
@click.option('--model_seq_length',
              default=256,
              type=click.IntRange(min=1),
              help=("Limits the length of the sequence returned. If tokenized string is shorter than max_length, "
                    "output will be padded with 0s. If the tokenized string is longer than max_length and "
                    "do_truncate == False, there will be multiple returned sequences containing the "
                    "overflowing token-ids. Default value is 256"))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@prepare_command()
def pipeline_nlp(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    \b
    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    \b
    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = Config.get()

    config.mode = PipelineModes.NLP

    config.feature_length = kwargs["model_seq_length"]

    from morpheus.pipeline import LinearPipeline

    ctx.obj = LinearPipeline(config)

    return ctx.obj


@click.group(chain=True, short_help="Run the inference pipeline with a FIL model", cls=AliasedGroup, **command_kwargs)
@click.option('--model_fea_length',
              default=29,
              type=click.IntRange(min=1),
              help="Number of features trained in the model")
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@prepare_command()
def pipeline_fil(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    \b
    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    \b
    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = Config.get()

    config.mode = PipelineModes.FIL

    config.feature_length = kwargs["model_fea_length"]

    from morpheus.pipeline import LinearPipeline

    ctx.obj = LinearPipeline(config)

    return ctx.obj


@click.pass_context
def post_pipeline(ctx: click.Context, stages, **kwargs):

    logger.info("Config: \n{}".format(Config.get().to_string()))

    click.secho("Starting pipeline via CLI... Ctrl+C to Quit", fg="red")

    from morpheus.pipeline import LinearPipeline

    pipeline: LinearPipeline = ctx.ensure_object(LinearPipeline)

    if ("viz_file" in kwargs and kwargs["viz_file"] is not None):
        pipeline.build()

        pipeline.visualize(kwargs["viz_file"], rankdir="LR")
        click.secho("Pipeline visualization saved to {}".format(kwargs["viz_file"]), fg="yellow")

    # Run the pipeline
    pipeline.run()


pipeline_nlp.result_callback = post_pipeline
pipeline_fil.result_callback = post_pipeline


@click.command(short_help="Load messages from a file", **command_kwargs)
@click.option('--filename', type=click.Path(exists=True, dir_okay=False), help="Input filename")
@prepare_command(False)
def from_file(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.input.from_file import FileSourceStage

    stage = FileSourceStage(Config.get(), **kwargs)

    p.set_source(stage)

    return stage


@click.command(short_help="Load messages from a Kafka cluster", **command_kwargs)
@click.option('--bootstrap_servers',
              type=str,
              default="auto",
              required=True,
              help=("Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, "
                    "this can be set to 'auto' to automatically determine the cluster IPs and ports"))
@click.option('--input_topic', type=str, default="test_pcap", required=True, help="Kafka topic to read from")
@click.option('--group_id', type=str, default="custreamz", required=True, help="")
@click.option('--use_dask', is_flag=True, help="Whether or not to use dask for multiple processes reading from Kafka")
@click.option('--poll_interval',
              type=str,
              default="10millis",
              required=True,
              help="Polling interval to check for messages. Follows the pandas interval format")
@prepare_command(False)
def from_kafka(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"] == "auto"):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from morpheus.pipeline.input.from_kafka import KafkaSourceStage

    stage = KafkaSourceStage(Config.get(), **kwargs)

    p.set_source(stage)

    return stage


@click.command(short_help="Display throughput numbers at a specific point in the pipeline", **command_kwargs)
@click.option('--description', type=str, required=True, help="Header message to use for this monitor")
@click.option('--smoothing',
              type=float,
              default=0.05,
              help="How much to average throughput numbers. 0=full average, 1=instantaneous")
@click.option('--unit', type=str, help="Units to use for data rate")
@prepare_command(False)
def monitor(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import MonitorStage

    stage = MonitorStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Buffer results", **command_kwargs)
@click.option('--count', type=int, default=1000, help="")
@prepare_command(False)
def buffer(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import BufferStage

    stage = BufferStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Delay results", **command_kwargs)
@click.option('--duration', type=str, help="Time to delay messages in the pipeline. Follows the pandas interval format")
@prepare_command(False)
def delay(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import DelayStage

    stage = DelayStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help=("Queue results until the previous stage is complete, then dump entire queue into pipeline. "
                           "Useful for testing stages independently. Requires finite source such as `from-file`"),
               **command_kwargs)
@prepare_command(False)
def trigger(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import TriggerStage

    stage = TriggerStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Deserialize source data from JSON", **command_kwargs)
@prepare_command(False)
def deserialize(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.preprocessing import DeserializeStage

    stage = DeserializeStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="preprocess", short_help="Convert messages to tokens", **command_kwargs)
@click.option('--vocab_hash_file',
              default="data/bert-base-cased-hash.txt",
              type=click.Path(exists=True, dir_okay=False),
              help=("Path to hash file containing vocabulary of words with token-ids. "
                    "This can be created from the raw vocabulary using the cudf.utils.hash_vocab_utils.hash_vocab "
                    "function. Default value is 'data/bert-base-cased-hash.txt'"))
@click.option('--truncation', default=False, type=bool)
@click.option('--do_lower_case', default=False, type=bool)
@click.option('--add_special_tokens', default=False, type=bool)
@click.option('--stride', type=int, default=-1, help="")
@prepare_command(False)
def preprocess_nlp(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.preprocessing import PreprocessNLPStage

    stage = PreprocessNLPStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="preprocess", short_help="Convert messages to tokens", **command_kwargs)
@prepare_command(False)
def preprocess_fil(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.preprocessing import PreprocessFILStage

    stage = PreprocessFILStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Perform inference with Triton", **command_kwargs)
@click.option('--model_name', type=str, required=True, help="Model name in Triton to send messages to")
@click.option('--server_url', type=str, required=True, help="Triton server URL (IP:Port)")
@click.option('--force_convert_inputs',
              default=False,
              type=bool,
              help=("Instructs this stage to forcibly convert all input types to match what Triton is expecting. "
                    "Even if this is set to `False`, automatic conversion will be done only if there would be no "
                    "data loss (i.e. int32 -> int64)."))
@prepare_command(False)
def inf_triton(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.inference.inference_triton import TritonInferenceStage

    stage = TritonInferenceStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Perform a no-op inference for testing", **command_kwargs)
@prepare_command(False)
def inf_identity(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.inference.inference_identity import IdentityInferenceStage

    stage = IdentityInferenceStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Perform inference with PyTorch", **command_kwargs)
@click.option('--model_filename',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="PyTorch model filename to load")
@prepare_command(False)
def inf_pytorch(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.inference.inference_pytorch import PyTorchInferenceStage

    stage = PyTorchInferenceStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Add detected classifications to each message", **command_kwargs)
@click.option('--threshold', type=float, default=0.5, required=True, help="Level to consider True/False")
@prepare_command(False)
def add_class(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import AddClassificationsStage

    if (Config.get().mode == PipelineModes.NLP):
        # Add the si_prefix to the classes
        kwargs["prefix"] = "si_"

        # Default labels
        kwargs["labels"] = [
            'address',
            'bank_acct',
            'credit_card',
            'email',
            'govt_id',
            'name',
            'password',
            'phone_num',
            'secret_keys',
            'user',
        ]
    elif (Config.get().mode == PipelineModes.FIL):
        # Only one class by default
        kwargs["labels"] = [
            "mining",
        ]

    stage = AddClassificationsStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Filter message by a classification threshold", **command_kwargs)
@click.option('--threshold', type=float, default=0.5, required=True, help="")
@prepare_command(False)
def filter(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.general_stages import FilterDetectionsStage

    stage = FilterDetectionsStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Deserialize source data from JSON", **command_kwargs)
@click.option('--include',
              type=str,
              default=tuple(),
              multiple=True,
              show_default="All Columns",
              help=("Which columns to include from MultiMessage into JSON. Can be specified multiple times. "
                    "Resulting columns is the intersection of all regex. Include applied before exclude"))
@click.option('--exclude',
              type=str,
              default=[r'^ID$', r'^ts_'],
              multiple=True,
              required=True,
              help=("Which columns to exclude from MultiMessage into JSON. Can be specified multiple times. "
                    "Resulting ignored columns is the intersection of all regex. Include applied before exclude"))
@prepare_command(False)
def serialize(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    kwargs["include"] = list(kwargs["include"])
    kwargs["exclude"] = list(kwargs["exclude"])

    from morpheus.pipeline.output.serialize import SerializeStage

    stage = SerializeStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Write all messages to a file", **command_kwargs)
@click.option('--filename', type=click.Path(writable=True), required=True, help="")
@click.option('--overwrite', is_flag=True, help="")
@prepare_command(False)
def to_file(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.output.to_file import WriteToFileStage

    stage = WriteToFileStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Write all messages to a Kafka cluster", **command_kwargs)
@click.option('--bootstrap_servers',
              type=str,
              default="auto",
              required=True,
              help=("Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, "
                    "this can be set to 'auto' to automatically determine the cluster IPs and ports"))
@click.option('--output_topic', type=str, required=True, help="Output Kafka topic to publish to")
@prepare_command(False)
def to_kafka(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"] == "auto"):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from morpheus.pipeline.output.to_kafka import WriteToKafkaStage

    stage = WriteToKafkaStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Write out vizualization data frames", **command_kwargs)
@click.option('--out_dir',
              type=click.Path(dir_okay=True, file_okay=False),
              default="./viz_frames",
              required=True,
              help="")
@click.option('--overwrite', is_flag=True, help="")
@prepare_command(False)
def gen_viz(ctx: click.Context, **kwargs):

    from morpheus.pipeline import LinearPipeline

    p: LinearPipeline = ctx.ensure_object(LinearPipeline)

    from morpheus.pipeline.output.gen_viz_frames import GenerateVizFramesStage

    stage = GenerateVizFramesStage(Config.get(), **kwargs)

    p.add_stage(stage)

    return stage


# Manually create the subcommands for each command (necessary since commands can be used on multiple groups)
run.add_command(dask)
run.add_command(pipeline_nlp)
run.add_command(pipeline_fil)

# Wrap both pipelines in dask commands
dask.add_command(pipeline_nlp)
dask.add_command(pipeline_fil)

# NLP Pipeline
pipeline_nlp.add_command(add_class)
pipeline_nlp.add_command(buffer)
pipeline_nlp.add_command(delay)
pipeline_nlp.add_command(deserialize)
pipeline_nlp.add_command(filter)
pipeline_nlp.add_command(from_kafka)
pipeline_nlp.add_command(from_file)
pipeline_nlp.add_command(gen_viz)
pipeline_nlp.add_command(inf_identity)
pipeline_nlp.add_command(inf_triton)
pipeline_nlp.add_command(inf_pytorch)
pipeline_nlp.add_command(monitor)
pipeline_nlp.add_command(preprocess_nlp)
pipeline_nlp.add_command(serialize)
pipeline_nlp.add_command(to_file)
pipeline_nlp.add_command(to_kafka)

# FIL Pipeline
pipeline_fil.add_command(add_class)
pipeline_fil.add_command(buffer)
pipeline_fil.add_command(delay)
pipeline_fil.add_command(deserialize)
pipeline_fil.add_command(filter)
pipeline_fil.add_command(from_kafka)
pipeline_fil.add_command(from_file)
pipeline_fil.add_command(inf_identity)
pipeline_fil.add_command(inf_triton)
pipeline_nlp.add_command(inf_pytorch)
pipeline_fil.add_command(monitor)
pipeline_fil.add_command(preprocess_fil)
pipeline_fil.add_command(serialize)
pipeline_fil.add_command(to_file)
pipeline_fil.add_command(to_kafka)


def run_cli():
    cli(obj={}, auto_envvar_prefix='CLX', show_default=True)


if __name__ == '__main__':
    run_cli()
