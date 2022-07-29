# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import collections
import importlib
import logging
import os
import types
import typing
import warnings
from functools import update_wrapper

import click
import pluggy
from click.globals import get_current_context

import morpheus
import morpheus.stages.input.cloud_trail_source_stage
from morpheus.cli import hookspecs
from morpheus.cli.default_command_hooks import DefaultCommandHooks
from morpheus.cli.utils import PluginSpec
from morpheus.cli.utils import _get_log_levels
from morpheus.cli.utils import _parse_log_level
from morpheus.cli.utils import get_config_from_ctx
from morpheus.cli.utils import get_pipeline_from_ctx
from morpheus.cli.utils import prepare_command
from morpheus.cli.utils import str_to_file_type
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import ConfigBase
from morpheus.config import ConfigFIL
from morpheus.config import ConfigOnnxToTRT
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.config import auto_determine_bootstrap
from morpheus.utils.logging import configure_logging

# pylint: disable=line-too-long, import-outside-toplevel, invalid-name, global-at-module-level, unused-argument

# NOTE: This file will get executed when hitting <TAB> for autocompletion. Any classes that require a long import time
# should be locally imported. For example, `morpheus.Pipeline` takes a long time to import and must be locally imported
# for each function

DEFAULT_CONFIG = Config()

# List all of the options in from morpheus._lib.file_types.FileTypes without importing the object. This slows down
# autocomplete too much.
FILE_TYPE_NAMES = ["auto", "csv", "json"]

command_kwargs = {
    "context_settings": dict(show_default=True, ),
}

ALIASES = {
    "pipeline": "pipeline-nlp",
}

global logger
logger = logging.getLogger("morpheus.cli")


class PluginManager():

    def __init__(self):
        self._pm = pluggy.PluginManager("morpheus")
        self._pm.add_hookspecs(hookspecs)
        self._pm.register(DefaultCommandHooks(), name="morpheus_default")

        self._plugins_loaded = False
        self._plugin_specs: typing.List[PluginSpec] = []

    def _get_plugin_specs_as_list(self, specs: PluginSpec) -> typing.List[str]:
        """Parse a plugins specification into a list of plugin names."""
        # None means empty.
        if specs is None:
            return []
        # Workaround for #3899 - a submodule which happens to be called "pytest_plugins".
        if isinstance(specs, types.ModuleType):
            return []
        # Comma-separated list.
        if isinstance(specs, str):
            return specs.split(",") if specs else []
        # Direct specification.
        if isinstance(specs, collections.abc.Sequence):
            return list(specs)
        raise RuntimeError("Plugins may be specified as a sequence or a ','-separated string of plugin names. Got: %r" %
                           specs)

    def _ensure_plugins_loaded(self):

        if (self._plugins_loaded):
            return

        # # Now that all command line plugins have been added, add any from the env variable
        self.add_plugin_option(os.environ.get("MORPHEUS_PLUGINS"))

        # Loop over all specs and load the plugins
        for s in self._plugin_specs:
            try:
                mod = importlib.import_module(s)

                # Sucessfully loaded. Register
                self._pm.register(mod)

            except ImportError as e:
                raise ImportError(f'Error importing plugin "{s}": {e.args[0]}').with_traceback(e.__traceback__) from e

        # Finally, consider setuptools entrypoints
        self._pm.load_setuptools_entrypoints("morpheus")

        self._plugins_loaded = True

    def add_plugin_option(self, spec: PluginSpec):
        # Append to the list of specs
        self._plugin_specs.extend(self._get_plugin_specs_as_list(spec))

    def get_pipeline_command_list(self, mode: PipelineModes) -> typing.List[str]:

        self._ensure_plugins_loaded()

        command_names = []

        # Run the plugins to find all commands
        returned_command_names_array = self._pm.hook.morpheus_cli_collect_stage_names(mode=mode)

        # Each plugin returns an array so convert the array of arrays to just an array
        for name_array in returned_command_names_array:
            command_names.extend(name_array)

        return command_names

    def get_pipeline_command(self, mode: PipelineModes, cmd_name: str) -> click.Command:

        self._ensure_plugins_loaded()

        command = self._pm.hook.morpheus_cli_make_stage_command(mode=mode, stage_name=cmd_name)

        return command


global PLUGIN_MANAGER
PLUGIN_MANAGER = None


def get_plugin_manager():
    global PLUGIN_MANAGER
    if (PLUGIN_MANAGER is None):
        PLUGIN_MANAGER = PluginManager()
    return PLUGIN_MANAGER


class AliasedGroup(click.Group):

    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name]
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)


class PluginGroup(AliasedGroup):

    def __init__(
        self,
        name: typing.Optional[str] = None,
        commands: typing.Optional[typing.Union[typing.Dict[str, click.Command], typing.Sequence[click.Command]]] = None,
        **attrs: typing.Any,
    ):
        self._pipeline_mode = attrs.pop("pipeline_mode", PipelineModes.OTHER)

        super().__init__(name, commands, **attrs)

        self._plugin_manager = get_plugin_manager()

    def list_commands(self, ctx: click.Context) -> typing.List[str]:

        # Get the list of commands from the base
        command_list = super().list_commands(ctx)

        # Extend it with any plugins
        plugin_command_list = self._plugin_manager.get_pipeline_command_list(self._pipeline_mode)

        duplicate_commands = [x for x in plugin_command_list if x in command_list]

        if (len(duplicate_commands) > 0):
            raise RuntimeError("Plugins registered the following duplicate commands: ".format(
                ", ".join(duplicate_commands)))

        command_list.extend(plugin_command_list)

        return command_list

    def get_command(self, ctx, cmd_name):

        # Check if the command is already loaded
        if (cmd_name not in self.commands):
            # Try and load it from the plugins
            self.commands[cmd_name] = self._plugin_manager.get_pipeline_command(self._pipeline_mode, cmd_name)

        return super().get_command(ctx, cmd_name)


class MorpheusRelativePath(click.Path):
    """
    A specialization of the `click.Path` class that falls back to using package relative paths if the file cannot be
    found. Takes the exact same parameters as `click.Path`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Append "data" to the name so it can be different than normal click.Path
        self.name = "data " + self.name

    def convert(self,
                value: typing.Any,
                param: typing.Optional["click.Parameter"],
                ctx: typing.Optional["click.Context"]) -> typing.Any:

        # First check if the path is relative
        if (not os.path.isabs(value)):

            # See if the file exists.
            does_exist = os.path.exists(value)

            if (not does_exist):
                # If it doesnt exist, then try to make it relative to the morpheus library root
                morpheus_root = os.path.dirname(morpheus.__file__)

                value_abs_to_root = os.path.join(morpheus_root, value)

                # If the file relative to our package exists, use that instead
                if (os.path.exists(value_abs_to_root)):
                    logger.debug(("Parameter, '%s', with relative path, '%s', does not exist. "
                                  "Using package relative location: '%s'"),
                                 param.name,
                                 value,
                                 value_abs_to_root)

                    return super().convert(value_abs_to_root, param, ctx)

        return super().convert(value, param, ctx)


@click.group(name="morpheus",
             chain=False,
             invoke_without_command=True,
             no_args_is_help=True,
             cls=AliasedGroup,
             **command_kwargs)
@click.option('--debug/--no-debug', default=False)
@click.option("--log_level",
              default=logging.getLevelName(DEFAULT_CONFIG.log_level),
              type=click.Choice(_get_log_levels(), case_sensitive=False),
              callback=_parse_log_level,
              help="Specify the logging level to use.")
@click.option('--log_config_file',
              default=DEFAULT_CONFIG.log_config_file,
              type=click.Path(exists=True, dir_okay=False),
              help=("Config file to use to configure logging. Use only for advanced situations. "
                    "Can accept both JSON and ini style configurations"))
@click.option(
    "plugins",
    '--plugin',
    #   envvar="PLUGINS",
    allow_from_autoenv=False,
    multiple=True,
    type=str,
    help="Adds a Morpheus CLI plugin. Can either be a module name or path to a python module")
@click.version_option()
@prepare_command(parse_config=True)
def cli(ctx: click.Context,
        log_level: int = DEFAULT_CONFIG.log_level,
        log_config_file: str = DEFAULT_CONFIG.log_config_file,
        plugins: typing.List[str] = None,
        **kwargs):

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    # Configure the logging
    configure_logging(log_level=log_level, log_config_file=log_config_file)

    # Re-get the logger class
    global logger  # pylint: disable=global-statement
    logger = logging.getLogger("morpheus.cli")

    if (plugins is not None):
        # If plugin is specified, add that to the plugin manager
        pm = get_plugin_manager()

        for p in plugins:
            pm.add_plugin_option(p)


@cli.group(short_help="Run a utility tool", no_args_is_help=True, **command_kwargs)
@prepare_command()
def tools(ctx: click.Context, **kwargs):

    pass


@tools.command(short_help="Converts an ONNX model to a TRT engine", **command_kwargs)
@click.option("--input_model", type=click.Path(exists=True, readable=True), required=True)
@click.option("--output_model", type=click.Path(exists=False, writable=True), required=True)
@click.option('--batches', type=(int, int), required=True, multiple=True)
@click.option('--seq_length', type=int, required=True)
@click.option('--max_workspace_size', type=int, default=16000)
@prepare_command()
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


@tools.group(short_help="Utility for installing/updating/removing shell completion for Morpheus",
             no_args_is_help=True,
             **command_kwargs)
def autocomplete(**kwargs):
    pass


@autocomplete.command(short_help="Show the Morpheus shell command completion code")
@click.option('--shell',
              required=False,
              type=click.Choice(["bash", "fish", "zsh"]),
              help="The shell to install completion to. Leave as the default to auto-detect")
def show(shell):
    """Show the click-completion-command completion code"""

    from morpheus.utils import click_completion_tools

    shell, path, code = click_completion_tools.get_code(shell=shell)

    click.secho("To add %s completion, write the following code to '%s':\n" % (shell, path), fg="blue")
    click.echo(code)


@autocomplete.command(short_help="Install the Morpheus shell command completion")
@click.option('--append/--overwrite', help="Append the completion code to the file", default=False)
@click.option('--shell',
              required=False,
              type=click.Choice(["bash", "fish", "zsh"]),
              help="The shell to install completion to. Leave as the default to auto-detect")
@click.option('--path',
              required=False,
              help="Location to install complete to. Leave empty to choose the default for the specified shell")
def install(**kwargs):
    """Install the click-completion-command completion"""

    from morpheus.utils import click_completion_tools

    shell, path = click_completion_tools.install_code(**kwargs)

    click.echo('%s completion installed in %s' % (shell, path))


@cli.group(short_help="Run one of the available pipelines", no_args_is_help=True, cls=AliasedGroup, **command_kwargs)
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
@click.option('--edge_buffer_size',
              default=DEFAULT_CONFIG.edge_buffer_size,
              type=click.IntRange(min=2),
              help=("The size of buffered channels to use between nodes in a pipeline. Larger values reduce "
                    "backpressure at the cost of memory. Smaller values will push messages through the "
                    "pipeline quicker. Must be greater than 1 and a power of 2 (i.e. 2, 4, 8, 16, etc.)"))
@click.option('--use_cpp',
              default=True,
              type=bool,
              help=("Whether or not to use C++ node and message types or to prefer python. "
                    "Only use as a last resort if bugs are encountered"))
@prepare_command(parse_config=True)
def run(ctx: click.Context, **kwargs):

    # Since the option isnt the same name as `should_use_cpp` anymore, manually set the value here.
    CppConfig.set_should_use_cpp(kwargs.pop("use_cpp", CppConfig.get_should_use_cpp()))

    pass


@click.group(chain=True,
             short_help="Run the inference pipeline with a NLP model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.NLP,
             **command_kwargs)
@click.option('--model_seq_length',
              default=256,
              type=click.IntRange(min=1),
              help=("Limits the length of the sequence returned. If tokenized string is shorter than max_length, "
                    "output will be padded with 0s. If the tokenized string is longer than max_length and "
                    "do_truncate == False, there will be multiple returned sequences containing the "
                    "overflowing token-ids. Default value is 256"))
@click.option('--labels_file',
              default="data/labels_nlp.txt",
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label"))
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

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.NLP
    config.feature_length = kwargs["model_seq_length"]

    if ("labels_file" in kwargs and kwargs["labels_file"] is not None):
        with open(kwargs["labels_file"], "r") as lf:
            config.class_labels = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run the inference pipeline with a FIL model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.FIL,
             **command_kwargs)
@click.option('--model_fea_length',
              default=29,
              type=click.IntRange(min=1),
              help="Number of features trained in the model")
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label. "
                    "If unspecified, only a single output label is created for FIL"))
@click.option('--columns_file',
              default="data/columns_fil.txt",
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read column features."))
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

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.FIL
    config.feature_length = kwargs["model_fea_length"]

    config.fil = ConfigFIL()

    if ("labels_file" in kwargs and kwargs["labels_file"] is not None):
        with open(kwargs["labels_file"], "r") as lf:
            config.class_labels = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))
    else:
        # Use a default single label
        config.class_labels = ["mining"]

    if ("columns_file" in kwargs and kwargs["columns_file"] is not None):
        with open(kwargs["columns_file"], "r") as lf:
            config.fil.feature_columns = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded columns. Current columns: [%s]", str(config.fil.feature_columns))
    else:
        raise ValueError('Unable to find columns file')

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run the inference pipeline with an AutoEncoder model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.AE,
             **command_kwargs)
@click.option('--columns_file',
              default="data/columns_ae.txt",
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=(""))
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label. "
                    "If unspecified, only a single output label is created for FIL"))
@click.option('--userid_column_name',
              type=str,
              default="userIdentityaccountId",
              required=True,
              help=("Which column to use as the User ID."))
@click.option('--userid_filter',
              type=str,
              default=None,
              help=("Specifying this value will filter all incoming data to only use rows with matching User IDs. "
                    "Which column is used for the User ID is specified by `userid_column_name`"))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@prepare_command()
def pipeline_ae(ctx: click.Context, **kwargs):
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

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.AE

    if CppConfig.get_should_use_cpp():
        logger.warning("C++ is disabled for AutoEncoder pipelines at this time.")
        CppConfig.set_should_use_cpp(False)

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = kwargs["userid_column_name"]

    if ("columns_file" in kwargs and kwargs["columns_file"] is not None):
        with open(kwargs["columns_file"], "r") as lf:
            config.ae.feature_columns = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded columns. Current columns: [%s]", str(config.ae.feature_columns))
    else:
        # Use a default single label
        config.class_labels = ["ae_anomaly_score"]

    if ("labels_file" in kwargs and kwargs["labels_file"] is not None):
        with open(kwargs["labels_file"], "r") as lf:
            config.class_labels = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))
    else:
        # Use a default single label
        config.class_labels = ["ae_anomaly_score"]

    if ("userid_filter" in kwargs):
        config.ae.userid_filter = kwargs["userid_filter"]

        logger.info("Filtering all users except ID: '%s'", str(config.ae.userid_filter))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run a custom inference pipeline without a specific model type",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.OTHER,
             **command_kwargs)
@click.option('--model_fea_length',
              default=1,
              type=click.IntRange(min=1),
              help="Number of features trained in the model")
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label."))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@prepare_command()
def pipeline_other(ctx: click.Context, **kwargs):
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

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.OTHER
    config.feature_length = kwargs["model_fea_length"]

    config.fil = ConfigFIL()

    if ("labels_file" in kwargs and kwargs["labels_file"] is not None):
        with open(kwargs["labels_file"], "r") as lf:
            config.class_labels = [x.strip() for x in lf.readlines()]
            logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@pipeline_nlp.result_callback()
@pipeline_fil.result_callback()
@pipeline_ae.result_callback()
@pipeline_other.result_callback()
@click.pass_context
def post_pipeline(ctx: click.Context, *args, **kwargs):

    config = get_config_from_ctx(ctx)

    logger.info("Config: \n%s", config.to_string())
    logger.info("CPP Enabled: %s", CppConfig.get_should_use_cpp())

    click.secho("Starting pipeline via CLI... Ctrl+C to Quit", fg="red")

    pipeline = get_pipeline_from_ctx(ctx)

    # Run the pipeline before generating visualization to ensure the pipeline has been started
    pipeline.run()

    # TODO(MDD): Move visualization before `pipeline.run()` once Issue #230 is fixed.
    if ("viz_file" in kwargs and kwargs["viz_file"] is not None):
        pipeline.visualize(kwargs["viz_file"], rankdir="LR")
        click.secho("Pipeline visualization saved to {}".format(kwargs["viz_file"]), fg="yellow")


@click.command(short_help="Load messages from a file", **command_kwargs)
@click.option('--filename', type=click.Path(exists=True, dir_okay=False), help="Input filename")
@click.option('--iterative',
              is_flag=True,
              default=False,
              help=("Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. "
                    "Iterative mode is good for interleaving source stages."))
@click.option('--file-type',
              type=click.Choice(FILE_TYPE_NAMES, case_sensitive=False),
              default="auto",
              help=("Indicates what type of file to read. "
                    "Specifying 'auto' will determine the file type from the extension."))
@click.option('--repeat',
              default=1,
              type=click.IntRange(min=1),
              help=("Repeats the input dataset multiple times. Useful to extend small datasets for debugging."))
@click.option('--filter_null',
              default=True,
              type=bool,
              help=("Whether or not to filter rows with null 'data' column. Null values in the 'data' column can "
                    "cause issues down the line with processing. Setting this to True is recommended."))
@prepare_command()
def from_file(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.input.file_source_stage import FileSourceStage

    file_type = str_to_file_type(kwargs.pop("file_type").lower())

    stage = FileSourceStage(config, file_type=file_type, **kwargs)

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
@click.option('--poll_interval',
              type=str,
              default="10millis",
              required=True,
              help="Polling interval to check for messages. Follows the pandas interval format")
@click.option("--disable_commit",
              is_flag=True,
              help=("Enabling this option will skip committing messages as they are pulled off the server. "
                    "This is only useful for debugging, allowing the user to process the same messages multiple times"))
@click.option("--disable_pre_filtering",
              is_flag=True,
              help=("Enabling this option will skip pre-filtering of json messages. "
                    "This is only useful when inputs are known to be valid json."))
@click.option("--auto_offset_reset",
              type=click.Choice(["earliest", "latest", "none"], case_sensitive=False),
              default="latest",
              help=("Sets the value for the configuration option 'auto.offset.reset'. "
                    "See the kafka documentation for more information on the effects of each value."))
@prepare_command()
def from_kafka(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"] == "auto"):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from morpheus.stages.input.kafka_source_stage import KafkaSourceStage

    stage = KafkaSourceStage(config, **kwargs)

    p.set_source(stage)

    return stage


# @click.command(short_help="Load messages from a Cloudtrail directory", **command_kwargs)
# @click.option('--input_glob',
#               type=str,
#               required=True,
#               help=("Input glob pattern to match files to read. For example, './input_dir/*.json' would read all "
#                     "files with the 'json' extension in the directory 'input_dir'."))
# @click.option('--watch_directory',
#               type=bool,
#               default=False,
#               help=("The watch directory option instructs this stage to not close down once all files have been read. "
#                     "Instead it will read all files that match the 'input_glob' pattern, and then continue to watch "
#                     "the directory for additional files. Any new files that are added that match the glob will then "
#                     "be processed."))
# @click.option('--max_files',
#               type=click.IntRange(min=1),
#               help=("Max number of files to read. Useful for debugging to limit startup time. "
#                     "Default value of -1 is unlimited."))
# @click.option('--file-type',
#               type=click.Choice(FILE_TYPE_NAMES, case_sensitive=False),
#               default="auto",
#               help=("Indicates what type of file to read. "
#                     "Specifying 'auto' will determine the file type from the extension."))
# @click.option('--repeat',
#               default=1,
#               type=click.IntRange(min=1),
#               help=("Repeats the input dataset multiple times. Useful to extend small datasets for debugging."))
# @click.option('--sort_glob',
#               type=bool,
#               default=False,
#               help=("If true the list of files matching `input_glob` will be processed in sorted order."))
# @click.option('--recursive',
#               type=bool,
#               default=True,
#               help=("If true, events will be emitted for the files in subdirectories matching `input_glob`."))
# @click.option('--queue_max_size',
#               type=int,
#               default=128,
#               help=("Maximum queue size to hold the file paths to be processed that match `input_glob`."))
# @click.option('--batch_timeout', type=float, default=5.0, help=("Timeout to retrieve batch messages from the queue."))
# @prepare_command()
# def from_cloudtrail(ctx: click.Context, **kwargs):

#     config = get_config_from_ctx(ctx)
#     p = get_pipeline_from_ctx(ctx)

#     from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage

#     file_type = str_to_file_type(kwargs.pop("file_type").lower())

#     stage = CloudTrailSourceStage(config, file_type=file_type, **kwargs)

#     p.set_source(stage)

#     return stage


@click.command(short_help="Display throughput numbers at a specific point in the pipeline", **command_kwargs)
@click.option('--description', type=str, required=True, help="Header message to use for this monitor")
@click.option('--smoothing',
              type=float,
              default=0.05,
              help="How much to average throughput numbers. 0=full average, 1=instantaneous")
@click.option('--unit', type=str, help="Units to use for data rate")
@click.option('--delayed_start',
              is_flag=True,
              help=("When delayed_start is enabled, the progress bar will not be shown until the first "
                    "message is received. Otherwise, the progress bar is shown on pipeline startup and "
                    "will begin timing immediately. In large pipelines, this option may be desired to "
                    "give a more accurate timing."))
@prepare_command()
def monitor(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.general.monitor_stage import MonitorStage

    stage = MonitorStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Buffer results", deprecated=True, **command_kwargs)
@click.option('--count', type=int, default=1000, help="")
@prepare_command()
def buffer(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.general.buffer_stage import BufferStage

    stage = BufferStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Drop null data entries from a DataFrame", **command_kwargs)
@click.option('--column', type=str, default="data", help="Which column to use when searching for null values.")
@prepare_command()
def dropna(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.drop_null_stage import DropNullStage

    stage = DropNullStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Buffer data until previous stage has completed",
               help=("This stage will buffer all inputs until the source stage is complete. At that point all messages "
                     "will be dumped into downstream stages. Useful for testing performance of one stage at a time."),
               **command_kwargs)
@prepare_command()
def trigger(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.general.trigger_stage import TriggerStage

    stage = TriggerStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Delay results for a certain duration", deprecated=True, **command_kwargs)
@click.option('--duration', type=str, help="Time to delay messages in the pipeline. Follows the pandas interval format")
@prepare_command()
def delay(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.general.delay_stage import DelayStage

    stage = DelayStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Deserialize source data from JSON.", **command_kwargs)
@prepare_command()
def deserialize(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

    stage = DeserializeStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Deserialize source data from JSON", **command_kwargs)
@click.option('--pretrained_filename',
              type=click.Path(exists=True, dir_okay=False),
              help=("Loads a single pre-trained model for all users."))
@click.option('--train_data_glob',
              type=str,
              help=("On startup, all files matching this glob pattern will be loaded and used "
                    "to train a model for each unique user ID."))
@click.option('--train_epochs',
              type=click.IntRange(min=1),
              default=25,
              help="The number of epochs to train user models for.")
@click.option('--train_max_history',
              type=click.IntRange(min=1),
              default=1000,
              help=("Maximum amount of rows that will be retained in history. As new data arrives, models will be "
                    "retrained with a maximum number of rows specified by this value."))
@click.option('--seed', type=int, default=None, help="Seed to use when training. Helps ensure consistent results.")
@prepare_command()
def train_ae(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.train_ae_stage import TrainAEStage

    stage = TrainAEStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="preprocess", short_help="Convert messages to tokens", **command_kwargs)
@click.option('--vocab_hash_file',
              default="data/bert-base-cased-hash.txt",
              type=MorpheusRelativePath(exists=True, dir_okay=False, resolve_path=True),
              help=("Path to hash file containing vocabulary of words with token-ids. "
                    "This can be created from the raw vocabulary using the cudf.utils.hash_vocab_utils.hash_vocab "
                    "function."))
@click.option('--truncation',
              default=False,
              type=bool,
              help=("When set to True, any tokens extending past the max sequence length will be truncated."))
@click.option('--do_lower_case', default=False, type=bool, help=("Converts all strings to lowercase."))
@click.option('--add_special_tokens',
              default=False,
              type=bool,
              help=("Adds special tokens '[CLS]' to the beginning and '[SEP]' to the end of each string. ."))
@click.option('--stride',
              type=int,
              default=-1,
              help=("If a string extends beyond max sequence length, it will be broken up into multiple sections. "
                    "This option specifies how far each to increment the head of each string when broken into "
                    "multiple segments. Lower numbers will reult in more overlap. Setting this to -1 will auto "
                    "calculate the stride to be 75% of the max sequence length. If truncation=True, "
                    "this option has no effect."))
@prepare_command()
def preprocess_nlp(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

    stage = PreprocessNLPStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="preprocess", short_help="Convert messages to tokens", **command_kwargs)
@prepare_command()
def preprocess_fil(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage

    stage = PreprocessFILStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="preprocess", short_help="Convert messages to tokens", **command_kwargs)
@prepare_command()
def preprocess_ae(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.preprocess.preprocess_ae_stage import PreprocessAEStage

    stage = PreprocessAEStage(config, **kwargs)

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
@click.option("--use_shared_memory",
              type=bool,
              default=False,
              help=("Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. "
                    "Using CUDA IPC reduces network transfer time but requires that Morpheus and Triton are "
                    "located on the same machine"))
@prepare_command()
def inf_triton(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage

    stage = TritonInferenceStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Perform a no-op inference for testing", **command_kwargs)
@prepare_command()
def inf_identity(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.inference.identity_inference_stage import IdentityInferenceStage

    stage = IdentityInferenceStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="inf-pytorch", short_help="Perform inference with PyTorch", **command_kwargs)
@click.option('--model_filename',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="PyTorch model filename to load")
@prepare_command()
def inf_pytorch(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.inference.pytorch_inference_stage import PyTorchInferenceStage

    stage = PyTorchInferenceStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="inf-pytorch", short_help="Perform inference with PyTorch", **command_kwargs)
@prepare_command()
def inf_pytorch_ae(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.inference.auto_encoder_inference_stage import AutoEncoderInferenceStage

    stage = AutoEncoderInferenceStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Add detected classifications to each message", **command_kwargs)
@click.option('--threshold', type=float, default=0.5, required=True, help="Level to consider True/False")
@click.option('--label',
              type=str,
              default=None,
              multiple=True,
              show_default="[Config.class_labels]",
              help=("Converts probability indexes into classification labels. If no labels are specified, "
                    "all labels from Config.class_labels will be added."))
@click.option('--prefix',
              type=str,
              default="",
              help=("Prefix to add to each label. Allows adding labels different from the "
                    "Config.class_labels property"))
@prepare_command()
def add_class(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    if ("label" in kwargs):
        if (kwargs["label"] is not None):
            # Convert to list named labels
            kwargs["labels"] = list(kwargs["label"])

        del kwargs["label"]

    from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage

    stage = AddClassificationsStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Add probability scores to each message", **command_kwargs)
@click.option('--label',
              type=str,
              default=None,
              multiple=True,
              show_default="[Config.class_labels]",
              help=("Converts probability indexes into scores. If no labels are specified, "
                    "all labels from Config.class_labels will be added."))
@click.option('--prefix',
              type=str,
              default="",
              help=("Prefix to add to each label. Allows adding labels different from the "
                    "Config.class_labels property"))
@prepare_command()
def add_scores(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    if ("label" in kwargs):
        if (kwargs["label"] is not None):
            # Convert to list named labels
            kwargs["labels"] = list(kwargs["label"])

        del kwargs["label"]

    from morpheus.stages.postprocess.add_scores_stage import AddScoresStage

    stage = AddScoresStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(name="filter", short_help="Filter message by a classification threshold", **command_kwargs)
@click.option('--threshold',
              type=float,
              default=0.5,
              required=True,
              help=("All messages without a probability above this threshold will be filtered away"))
@prepare_command()
def filter_command(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage

    stage = FilterDetectionsStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Include & exclude columns from messages", **command_kwargs)
@click.option('--include',
              type=str,
              default=tuple(),
              multiple=True,
              show_default="All Columns",
              help=("Which columns to include from MultiMessage into JSON. Can be specified multiple times. "
                    "Resulting columns is the intersection of all regex. Include applied before exclude"))
@click.option('--exclude',
              type=str,
              default=[r'^ID$', r'^_ts_'],
              multiple=True,
              required=True,
              help=("Which columns to exclude from MultiMessage into JSON. Can be specified multiple times. "
                    "Resulting ignored columns is the intersection of all regex. Include applied before exclude"))
@prepare_command()
def serialize(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    kwargs["include"] = list(kwargs["include"])
    kwargs["exclude"] = list(kwargs["exclude"])

    from morpheus.stages.postprocess.serialize_stage import SerializeStage

    stage = SerializeStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Report model drift statistics to ML Flow", **command_kwargs)
@click.option('--tracking_uri',
              type=str,
              default=None,
              help=("The ML Flow tracking URI to connect to the tracking backend. If not speficied, MF Flow will use "
                    "'file:///mlruns' relative to the current directory"))
@click.option('--experiment_name', type=str, default="Morpheus", help=("The experiement name to use in ML Flow"))
@click.option('--run_id',
              type=str,
              default=None,
              help=("The ML Flow Run ID to report metrics to. If unspecified, Morpheus will attempt to reuse any "
                    "previously created runs that are still active. Otherwise, a new run will be created. By default, "
                    "runs are left in an active state."))
@click.option('--labels',
              type=str,
              default=tuple(),
              multiple=True,
              show_default="Determined by mode",
              help=("Converts probability indexes into labels for the ML Flow UI. If no labels are specified, "
                    "the probability labels are determined by the pipeline mode."))
@click.option('--batch_size',
              type=int,
              default=-1,
              help=("The batch size to calculate model drift statistics. Allows for increasing or decreasing how "
                    "much data is reported to MLFlow. Default is -1 which will use the pipeline batch_size."))
@click.option('--force_new_run',
              is_flag=True,
              help=("Whether or not to reuse the most recent run ID in ML Flow or create a new one each time the "
                    "pipeline is run"))
@prepare_command()
def mlflow_drift(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    # Ensure labels is not a tuple
    kwargs["labels"] = list(kwargs["labels"])

    from morpheus.stages.postprocess.ml_flow_drift_stage import MLFlowDriftStage

    stage = MLFlowDriftStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Perform time series anomaly detection and add prediction.", **command_kwargs)
@click.option('--resolution',
              type=str,
              default="1 h",
              help=("Time series resolution. Logs will be binned into groups of this size. Uses the pandas time "
                    "delta format, i.e. '10m' for 10 minutes"))
@click.option('--min_window',
              type=str,
              default="12 h",
              help=("Minimum window on either side of a log necessary for calculation. Logs will be skipped "
                    "during a warmup phase while this window is filled. Uses the pandas time delta format, "
                    "i.e. '10m' for 10 minutes"))
@click.option('--hot_start',
              is_flag=True,
              default=False,
              help=("This flag prevents the stage from ignoring messages during a warm up phase while the "
                    "min_window is filled. Enabling 'hot_start' will run calculations on all messages even "
                    "if the min_window is not satisfied on both sides, i.e. during startup or teardown. This "
                    "is likely to increase the number of false positives but can be helpful for debugging "
                    "and testing on small datasets."))
@click.option('--cold_end',
              is_flag=True,
              default=False,
              help=("This flag prevents the stage from ignoring messages during a warm up phase while the "
                    "min_window is filled. Enabling 'hot_start' will run calculations on all messages even "
                    "if the min_window is not satisfied on both sides, i.e. during startup or teardown. This "
                    "is likely to increase the number of false positives but can be helpful for debugging "
                    "and testing on small datasets."))
@click.option('--filter_percent',
              type=click.FloatRange(min=0.0, max=100.0),
              default=90.0,
              required=True,
              help="The percent of timeseries samples to remove from the inverse FFT for spectral density filtering.")
@click.option('--zscore_threshold',
              type=click.FloatRange(min=0.0),
              default=8.0,
              required=True,
              help=("The z-score threshold required to flag datapoints. The value indicates the number of standard "
                    "deviations from the mean that is required to be flagged. Increasing this value will decrease "
                    "the number of detections."))
@prepare_command()
def timeseries(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage

    stage = TimeSeriesStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Validates pipeline output against an expected output", **command_kwargs)
@click.option('--val_file_name',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help=("File to use as the comparison 'truth' object. CSV files are preferred"))
@click.option('--results_file_name',
              type=click.Path(dir_okay=False),
              required=True,
              help=("Output filename to store a JSON object containing the validation results"))
@click.option('--overwrite', is_flag=True, help=("Whether or not to overwrite existing JSON output"))
@click.option('--include',
              type=str,
              default=tuple(),
              multiple=True,
              show_default="All Columns",
              help=("Which columns to include in the validation. Can be specified multiple times. "
                    "Resulting columns is the intersection of all regex. Include applied before exclude"))
@click.option('--exclude',
              type=str,
              default=[r'^ID$', r'^_ts_'],
              multiple=True,
              required=True,
              help=("Which columns to exclude from the validation. Can be specified multiple times. "
                    "Resulting ignored columns is the intersection of all regex. Include applied before exclude"))
@click.option('--index_col',
              type=str,
              help=("Specifies a column which will be used to align messages with rows in the validation dataset."))
@click.option('--abs_tol',
              type=click.FloatRange(min=0.0),
              default=0.001,
              required=True,
              help="Absolute tolerance to use when comparing float columns.")
@click.option('--rel_tol',
              type=click.FloatRange(min=0.0),
              default=0.05,
              required=True,
              help="Relative tolerance to use when comparing float columns.")
@prepare_command()
def validate(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.postprocess.validation_stage import ValidationStage

    stage = ValidationStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Write all messages to a file", **command_kwargs)
@click.option('--filename', type=click.Path(writable=True), required=True, help="The file to write to")
@click.option('--overwrite', is_flag=True, help="Whether or not to overwrite the target file")
@prepare_command()
def to_file(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.output.write_to_file_stage import WriteToFileStage

    stage = WriteToFileStage(config, **kwargs)

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
@prepare_command()
def to_kafka(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    if ("bootstrap_servers" in kwargs and kwargs["bootstrap_servers"] == "auto"):
        kwargs["bootstrap_servers"] = auto_determine_bootstrap()

    from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage

    stage = WriteToKafkaStage(config, **kwargs)

    p.add_stage(stage)

    return stage


@click.command(short_help="Write out vizualization data frames", deprecated=True, **command_kwargs)
@click.option('--out_dir',
              type=click.Path(dir_okay=True, file_okay=False),
              default="./viz_frames",
              required=True,
              help="")
@click.option('--overwrite', is_flag=True, help="")
@prepare_command()
def gen_viz(ctx: click.Context, **kwargs):

    config = get_config_from_ctx(ctx)
    p = get_pipeline_from_ctx(ctx)

    from morpheus.stages.postprocess.generate_viz_frames_stage import GenerateVizFramesStage

    stage = GenerateVizFramesStage(config, **kwargs)

    p.add_stage(stage)

    return stage


# Manually create the subcommands for each command (necessary since commands can be used on multiple groups)
run.add_command(pipeline_nlp)
run.add_command(pipeline_fil)
run.add_command(pipeline_ae)
run.add_command(pipeline_other)

# NLP Pipeline
pipeline_nlp.add_command(add_class)
pipeline_nlp.add_command(add_scores)
pipeline_nlp.add_command(buffer)
pipeline_nlp.add_command(delay)
pipeline_nlp.add_command(deserialize)
pipeline_nlp.add_command(dropna)
pipeline_nlp.add_command(filter_command)
pipeline_nlp.add_command(from_file)
pipeline_nlp.add_command(from_kafka)
pipeline_nlp.add_command(gen_viz)
pipeline_nlp.add_command(inf_identity)
pipeline_nlp.add_command(inf_pytorch)
pipeline_nlp.add_command(inf_triton)
pipeline_nlp.add_command(mlflow_drift)
pipeline_nlp.add_command(monitor)
pipeline_nlp.add_command(preprocess_nlp)
pipeline_nlp.add_command(serialize)
pipeline_nlp.add_command(to_file)
pipeline_nlp.add_command(to_kafka)
pipeline_nlp.add_command(validate)

# FIL Pipeline
pipeline_fil.add_command(add_class)
pipeline_fil.add_command(add_scores)
pipeline_fil.add_command(buffer)
pipeline_fil.add_command(delay)
pipeline_fil.add_command(deserialize)
pipeline_fil.add_command(dropna)
pipeline_fil.add_command(filter_command)
pipeline_fil.add_command(from_file)
pipeline_fil.add_command(from_kafka)
pipeline_fil.add_command(inf_identity)
pipeline_fil.add_command(inf_pytorch)
pipeline_fil.add_command(inf_triton)
pipeline_fil.add_command(mlflow_drift)
pipeline_fil.add_command(monitor)
pipeline_fil.add_command(preprocess_fil)
pipeline_fil.add_command(serialize)
pipeline_fil.add_command(to_file)
pipeline_fil.add_command(to_kafka)
pipeline_fil.add_command(validate)

# AE Pipeline
pipeline_ae.add_command(add_class)
pipeline_ae.add_command(add_scores)
pipeline_ae.add_command(buffer)
pipeline_ae.add_command(delay)
pipeline_ae.add_command(filter_command)
# pipeline_ae.add_command(from_cloudtrail)
pipeline_ae.add_command(gen_viz)
pipeline_ae.add_command(inf_pytorch_ae)
pipeline_ae.add_command(inf_triton)
pipeline_ae.add_command(monitor)
pipeline_ae.add_command(preprocess_ae)
pipeline_ae.add_command(serialize)
pipeline_ae.add_command(timeseries)
pipeline_ae.add_command(to_file)
pipeline_ae.add_command(to_kafka)
pipeline_ae.add_command(train_ae)
pipeline_ae.add_command(validate)

# Other Pipeline
pipeline_other.add_command(add_class)
pipeline_other.add_command(add_scores)
pipeline_other.add_command(buffer)
pipeline_other.add_command(delay)
pipeline_other.add_command(deserialize)
pipeline_other.add_command(dropna)
pipeline_other.add_command(filter_command)
pipeline_other.add_command(from_file)
pipeline_other.add_command(from_kafka)
pipeline_other.add_command(inf_identity)
pipeline_other.add_command(inf_pytorch)
pipeline_other.add_command(inf_triton)
pipeline_other.add_command(mlflow_drift)
pipeline_other.add_command(monitor)
pipeline_other.add_command(serialize)
pipeline_other.add_command(to_file)
pipeline_other.add_command(to_kafka)
pipeline_other.add_command(validate)


def run_cli():
    print("Running CLI")
    cli(obj={}, auto_envvar_prefix='MORPHEUS', show_default=True, prog_name="morpheus")


if __name__ == '__main__':
    run_cli()
