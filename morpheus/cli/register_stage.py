import enum
import inspect
import pathlib
import re
import typing
from enum import Enum

import click
import numpydoc.docscrape
from typing_utils import get_args
from typing_utils import issubtype

from morpheus.cli.stage_registry import GlobalStageRegistry
from morpheus.cli.stage_registry import LazyStageInfo
from morpheus.cli.stage_registry import StageInfo
from morpheus.cli.stage_registry import StageRegistry
from morpheus.cli.utils import get_config_from_ctx
from morpheus.cli.utils import get_pipeline_from_ctx
from morpheus.cli.utils import prepare_command
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import get_full_qualname

# def _get_config_from_ctx(ctx) -> Config:
#     ctx_dict = ctx.ensure_object(dict)

#     if "config" not in ctx_dict:
#         ctx_dict["config"] = Config()

#     return ctx_dict["config"]

# def _get_pipeline_from_ctx(ctx):
#     ctx_dict = ctx.ensure_object(dict)

#     assert "pipeline" in ctx_dict, "Inconsistent configuration. Pipeline accessed before created"

#     return ctx_dict["pipeline"]

# def _prepare_command(parse_config: bool = False):

#     def inner_prepare_command(f):
#         """
#         Preparse command for use. Combines @without_empty_args, @show_defaults and @click.pass_context
#         """

#         def new_func(*args, **kwargs):
#             ctx: click.Context = get_current_context()
#             ctx.show_default = True

#             kwargs = _without_empty_args(kwargs)

#             # Apply the config if desired
#             if parse_config:
#                 config = get_config_from_ctx(ctx)

#                 _apply_to_config(config, **kwargs)

#             return f(ctx, *args, **kwargs)

#         return update_wrapper(new_func, f)

#     return inner_prepare_command


def class_name_to_command_name(class_name: str) -> str:
    # Convert to snake case with dashes
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '-', class_name).lower()

    # remove trailing "stage"
    return re.sub(r'-stage$', '', snake_case)


def get_param_doc(numpydoc_obj: numpydoc.docscrape.NumpyDocString, name: str):
    found_doc = next((x for x in numpydoc_obj["Parameters"] if x.name == name), None)

    if (not found_doc):
        return ""

    # Combine the docstrings into a single string
    param_doc = " ".join(found_doc.desc)

    return param_doc


def get_param_type(numpydoc_obj: numpydoc.docscrape.NumpyDocString, name: str):
    found_doc = next((x for x in numpydoc_obj["Parameters"] if x.name == name), None)

    if (not found_doc):
        return ""

    return found_doc.type


def parse_type_value(value_str: str) -> typing.Any:

    value_lower = value_str.lower()

    # Check for bool
    if (value_lower == "true"):
        return True
    if (value_lower == "false"):
        return False
    if (value_str.startswith('"') and value_str.endswith('"')):
        return value_str
    if (value_str.startswith("'") and value_str.endswith("'")):
        return value_str

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Otherwise just return none
    return None


def parse_doc_type_str(doc_type_str: str) -> dict:

    # Split along ,
    comma_split = doc_type_str.split(",")

    out_dict = {}

    # Now loop over comma separated
    for x in comma_split:
        # Split on =
        equal_split = x.split("=")

        if (len(equal_split) == 2):
            # Key value pair
            out_dict[equal_split[0].strip()] = parse_type_value(equal_split[1].strip())
        elif (len(equal_split) == 1):
            # Single type
            out_dict[""] = equal_split[0].strip()
        else:
            raise RuntimeError("Invalid docstring: {}".format(doc_type_str))

    return out_dict


def get_doc_kwargs(doc_type_str: str) -> dict:

    out_dict = parse_doc_type_str(doc_type_str)

    # pop out the empty and default
    out_dict.pop("", None)
    out_dict.pop("default", None)

    return out_dict


def set_options_param_type(options_kwargs: dict, annotation, doc_type: str):

    if (annotation == inspect.Parameter.empty):
        raise RuntimeError("All types must be specified to auto register stage")

    if (issubtype(annotation, typing.List)):
        # For variable length array, use multiple=True
        options_kwargs["multiple"] = True
        options_kwargs["type"] = get_args(annotation)[0]
    elif (issubtype(annotation, pathlib.Path)):

        type_dict = get_doc_kwargs(doc_type)

        # For paths, use the Path option and apply any kwargs
        options_kwargs["type"] = click.Path(**type_dict)
    elif (issubtype(annotation, Enum)):
        options_kwargs["type"] = click.Choice()

    elif (issubtype(annotation, bool)):

        options_kwargs["type"] = annotation

        type_dict = get_doc_kwargs(doc_type)

        # Apply the is_flag option
        if (type_dict.get("is_flag", False)):
            options_kwargs["is_flag"] = True

    else:
        options_kwargs["type"] = annotation


def register_stage(command_name: str = None,
                   modes: typing.Sequence[PipelineModes] = None,
                   ignore_args: typing.List[str] = list(),
                   command_args: dict = dict(),
                   option_args: typing.Dict[str, dict] = dict()):

    if (modes is None):
        modes = [x for x in PipelineModes]

    def register_stage_inner(stage_class: _DecoratorType) -> _DecoratorType:

        nonlocal command_name

        if (not hasattr(stage_class, "_morpheus_registered_stage")):

            # Determine the command name if it wasnt supplied
            if (command_name is None):
                command_name = class_name_to_command_name(stage_class.__name__)

            def build_command():

                command_params: typing.List[click.Parameter] = []

                class_init_sig = inspect.signature(stage_class)

                config_param_name = None

                numpy_doc: numpydoc.docscrape.NumpyDocString = None

                # Try to get the docstring from __init__first, then from the class itself
                if (hasattr(stage_class, "__init__") and stage_class.__init__.__doc__ is not None):
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class.__init__)
                else:
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class)

                for p_name, p_value in class_init_sig.parameters.items():

                    if (p_value.annotation == Config):
                        config_param_name = p_name
                        continue
                    elif (p_name in ignore_args):
                        assert p_value.default != inspect.Parameter.empty, "Cannot ignore argument without default value"
                        continue
                    elif (p_value.kind == inspect.Parameter.VAR_POSITIONAL):
                        continue
                    elif (p_value.kind == inspect.Parameter.VAR_KEYWORD):
                        continue

                    option_kwargs = {}
                    option_kwargs["required"] = True

                    # See if we have some sort of documentation for this argument
                    option_kwargs["help"] = get_param_doc(numpy_doc, p_name)

                    # Set the default value if not empty
                    option_kwargs["default"] = p_value.default if p_value.default != inspect.Parameter.empty else None

                    set_options_param_type(option_kwargs, p_value.annotation, get_param_type(numpy_doc, p_name))

                    # Now overwrite with any user supplied values for this option
                    option_kwargs.update(option_kwargs.get(p_name, {}))

                    option = click.Option((f"--{p_name}", ), **option_kwargs)

                    command_params.append(option)

                if (config_param_name is None):
                    raise RuntimeError("All stages must take on argument of morpheus.Config. Ensure your stage "
                                       "constructor as one argument that has been annotated with morpheus.Config. "
                                       "i.e. `c: morpheus.Config`")

                def command_callback(ctx: click.Context, **kwargs):

                    # Delay loading SourceStage
                    from morpheus.pipeline.source_stage import SourceStage

                    config = get_config_from_ctx(ctx)
                    p = get_pipeline_from_ctx(ctx)

                    # Set the config to the correct parameter
                    kwargs[config_param_name] = config

                    stage = stage_class(**kwargs)

                    if (issubclass(stage_class, SourceStage)):
                        p.set_source(stage)
                    else:
                        p.add_stage(stage)

                    return stage

                command_kwargs = {}

                command_kwargs["short_help"] = "\n".join(numpy_doc["Summary"] or [])
                command_kwargs["help"] = "\n".join(numpy_doc["Extended Summary"] or [command_kwargs["short_help"]])

                # Now overwrite with any user supplied values
                command_kwargs.update(command_args)

                command = click.Command(name=command_name,
                                        callback=prepare_command()(command_callback),
                                        params=command_params,
                                        **command_kwargs)

                return command

            # Create the StageInfo
            stage_info = StageInfo(name=command_name, modes=modes, build_command=build_command)

            # Save the stage information to be retrieved later
            stage_class._morpheus_registered_stage = stage_info

            existing_registrations: typing.Set[PipelineModes] = set()

            # Get any already registered nodes
            for m in modes:
                registered_stage = GlobalStageRegistry.get().get_stage_info(command_name, m)

                if (registered_stage is not None):
                    # Verify its a lazy stage info with the same
                    if (not isinstance(registered_stage, LazyStageInfo)):
                        raise RuntimeError(
                            "Registering stage '{}' failed. Stage is already registered. Ensure `register_stage` is only executed once for each mode and name combination"
                        )

                    if (registered_stage.qualified_name != get_full_qualname(stage_class)):
                        raise RuntimeError("")

                    existing_registrations.update(registered_stage.modes)

            if (len(existing_registrations) > 0):
                diff_modes = existing_registrations.difference(modes)
                if (len(diff_modes) > 0):
                    raise RuntimeError("Mismatch between LazyStageInfo modes and register_stage() modes")
            else:
                # Not registered, add to global registry
                GlobalStageRegistry.get().add_stage_info(stage_info)

        return stage_class

    return register_stage_inner
