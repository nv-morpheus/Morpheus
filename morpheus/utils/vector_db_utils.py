# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import importlib
import inspect

from pymilvus import DataType

from morpheus.controllers.vector_db_controller import VectorDBController

MILVUS_DATA_TYPE_MAP = {
    "int64": DataType.INT64,
    "bool": DataType.BOOL,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "binary_vector": DataType.BINARY_VECTOR,
    "float_vector": DataType.FLOAT_VECTOR
}


def with_mutex(lock_name):
    """
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            with getattr(args[0], lock_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def load_handler_by_path(handler_class_path, kwargs):
    """
    Dynamically loads and instantiates a handler class specified by its import path.

    Parameters
    ----------
    handler_class_path : str
        The import path to the handler class to be loaded.
    kwargs : dict
        A dictionary of keyword arguments to pass to the handler class constructor.

    Returns
    -------
    VectorDatabaseHandler
        An instance of the loaded handler class.

    Raises
    ------
    ImportError
        If the specified module cannot be imported.
    AttributeError
        If no classes that inherit from VectorDatabaseHandler are found in the module.
    ValueError
        If required constructor keyword arguments are missing or if no valid custom handler is found.
    """

    try:
        module = importlib.import_module(handler_class_path)
        for obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, VectorDBController) and obj != VectorDBController:
                handler_constructor_args = inspect.signature(obj.__init__).parameters.keys()

                # Validate constructor keyword arguments
                missing_kwargs = {kwarg for kwarg in handler_constructor_args if kwarg not in kwargs}
                if missing_kwargs:
                    raise ValueError(
                        f"Missing required constructor keyword arguments for '{obj.__name__}': {', '.join(missing_kwargs)}"
                    )

                return obj(**kwargs)  # Instantiate with kwargs
        raise ValueError(f"No valid custom handler found in the specified class path '{handler_class_path}'")
    except ImportError:
        raise ImportError(
            f"Failed to import module for the specified class path '{handler_class_path}'. Make sure the module exists."
        )
    except AttributeError:
        raise AttributeError(
            f"No classes that inherit from VectorDatabaseHandler found in module for the specified class path '{handler_class_path}'"
        )
