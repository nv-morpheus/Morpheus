import inspect
import re
import typing

import click
import numpydoc.docscrape
from typing_utils import get_args
from typing_utils import issubtype

from morpheus.cli.stage_registry import StageInfo
from morpheus.cli.stage_registry import StageRegistry
from morpheus.cli.utils import get_config_from_ctx
from morpheus.cli.utils import get_pipeline_from_ctx
from morpheus.cli.utils import prepare_command
from morpheus.config import Config
from morpheus.config import PipelineModes

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


def set_options_param_type(options_kwargs: dict, annotation):

    if (annotation == inspect.Parameter.empty):
        raise RuntimeError("All types must be specified to auto register stage")

    if (issubtype(annotation, typing.List)):
        # For variable length array, use multiple=True
        options_kwargs["multiple"] = True
        options_kwargs["type"] = get_args(annotation)[0]

    else:
        options_kwargs["type"] = annotation


def register_stage(command_name: str = None, modes: typing.Sequence[PipelineModes] = None):

    def register_stage_inner(stage_class):

        nonlocal command_name

        if (not hasattr(stage_class, "_registered_with_morpheus")):

            # Determine the command name if it wasnt supplied
            if (command_name is None):
                command_name = class_name_to_command_name(stage_class.__name__)

            def build_command():

                command_params: typing.List[click.Parameter] = []

                class_init_sig = inspect.signature(stage_class)

                config_param_name = None

                numpy_doc: numpydoc.docscrape.NumpyDocString = None

                # Try to get the docstring from __init__first, then from the class itself
                if (hasattr(stage_class, "__init__") and inspect.getdoc(stage_class.__init__) is not None):
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class.__init__)
                else:
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class)

                for p_name, p_value in class_init_sig.parameters.items():

                    if (p_value.annotation == Config):
                        config_param_name = p_name
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

                    set_options_param_type(option_kwargs, p_value.annotation)

                    option = click.Option((f"--{p_name}", ), **option_kwargs)

                    command_params.append(option)

                if (config_param_name is None):
                    raise RuntimeError("All stages must take on argument of morpheus.Config. Ensure your stage "
                                       "constructor as one argument that has been annotated with morpheus.Config. "
                                       "i.e. `c: morpheus.Config`")

                def command_callback(ctx: click.Context, **kwargs):
                    config = get_config_from_ctx(ctx)
                    p = get_pipeline_from_ctx(ctx)

                    # Set the config to the correct parameter
                    kwargs[config_param_name] = config

                    stage = stage_class(**kwargs)

                    p.add_stage(stage)

                    return stage

                command = click.Command(name=command_name,
                                        callback=prepare_command()(command_callback),
                                        params=command_params)

                return command

            # Create the StageInfo
            stage_info = StageInfo(command_name, build_command, modes)

            StageRegistry.add_stage_info(stage_info)

            stage_class._registered_with_morpheus = True

        return stage_class

    return register_stage_inner
