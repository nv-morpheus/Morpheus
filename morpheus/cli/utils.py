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

import logging
import os
import types
import typing
import warnings
from functools import update_wrapper

import click
import click.globals

import morpheus
from morpheus.config import Config
from morpheus.config import ConfigBase

# Ignore pipeline unless we are typechecking since it takes a while to import
if (typing.TYPE_CHECKING):
    from morpheus.pipeline.linear_pipeline import LinearPipeline

logger = logging.getLogger(__name__)

PluginSpec = typing.Union[None, types.ModuleType, str, typing.Sequence[str]]


def str_to_file_type(file_type_str: str):
    # Delay FileTypes since this will import ._lib
    from morpheus._lib.file_types import FileTypes
    file_type_members = {name.lower(): t for (name, t) in FileTypes.__members__.items()}

    return file_type_members[file_type_str]


def _without_empty_args(passed_args):
    return {k: v for k, v in passed_args.items() if v is not None}


def without_empty_args(f):
    """
    Removes keyword arguments that have a None value
    """

    def new_func(*args, **kwargs):
        kwargs = _without_empty_args(kwargs)
        return f(click.globals.get_current_context(), *args, **kwargs)

    return update_wrapper(new_func, f)


def show_defaults(f):
    """
    Ensures the click.Context has `show_defaults` set to True. (Seems like a bug currently)
    """

    def new_func(*args, **kwargs):
        ctx: click.Context = click.globals.get_current_context()
        ctx.show_default = True
        return f(*args, **kwargs)

    return update_wrapper(new_func, f)


def _apply_to_config(config: ConfigBase, **kwargs):
    for param in kwargs:
        if hasattr(config, param):
            setattr(config, param, kwargs[param])
        else:
            warnings.warn(f"No config option matches for {param}")

    return config


def prepare_command(parse_config: bool = False):

    def inner_prepare_command(f):
        """
        Preparse command for use. Combines @without_empty_args, @show_defaults and @click.pass_context
        """

        def new_func(*args, **kwargs):
            ctx: click.Context = click.globals.get_current_context()
            ctx.show_default = True

            # Set the max width. This will still default to the users console width
            ctx.max_content_width = 200

            kwargs = _without_empty_args(kwargs)

            # Apply the config if desired
            if parse_config:
                config = get_config_from_ctx(ctx)

                _apply_to_config(config, **kwargs)

            return f(ctx, *args, **kwargs)

        return update_wrapper(new_func, f)

    return inner_prepare_command


def get_config_from_ctx(ctx) -> Config:
    ctx_dict = ctx.ensure_object(dict)

    if "config" not in ctx_dict:
        ctx_dict["config"] = Config()

    return ctx_dict["config"]


def get_pipeline_from_ctx(ctx) -> "LinearPipeline":
    ctx_dict = ctx.ensure_object(dict)

    assert "pipeline" in ctx_dict, "Inconsistent configuration. Pipeline accessed before created"

    return ctx_dict["pipeline"]


morpheus_log_levels = list(logging._nameToLevel.keys())

if ("NOTSET" in morpheus_log_levels):
    morpheus_log_levels.remove("NOTSET")


def get_log_levels():
    """
    Returns a list of all available logging levels in string format
    """
    return morpheus_log_levels


def parse_log_level(ctx, param, value):
    """
    Click callback that parses a command line value into a logging level
    """

    x = logging._nameToLevel.get(value.upper(), None)
    if x is None:
        raise click.BadParameter('Must be one of {}. Passed: {}'.format(", ".join(logging._nameToLevel.keys()), value))
    return x


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
