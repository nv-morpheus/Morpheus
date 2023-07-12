# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import inspect
import typing

# def execution_chain_annotations(*annotations):
#    def inner_annotate(func):
#        def wrapper(*args, **kwargs):
#            _annocations = list(*annotations)
#
#            result = func(*args, **kwargs)
#            signature = inspect.signature(func)
#
#            return {key: value for key, value in result.items() if key in signature.return_annotation}
#
#        return wrapper
#
#    return inner_annotate
#
#
# @execution_chain_annotations("x", "y", "z")
# def myfunc(a, b, c) -> (int, int, int):
#    x, y, z = 1, 2, 3
#
#    return x, y, z


class ExecutionChain:
    """
    A class that represents a chain of functions to be executed sequentially.

    Attributes
    ----------
    functions : List[Callable]
        List of functions to be executed in the chain.

    Methods
    -------
    __call__(**kwargs):
        Executes all the functions in the chain.
    add_function(function: Callable, position: int = -1):
        Adds a function to the chain at a specified position.
    remove_function(function: Callable):
        Removes a specific function from the chain.
    replace_function(old_function: Callable, new_function: Callable):
        Replaces a specific function in the chain with a new one.
    validate_chain():
        Validates the function chain.
    """

    def __init__(self, function_chain: typing.Optional[typing.List[typing.Callable]] = None):
        """
        Constructs all the necessary attributes for the ExecutionChain object.

        Parameters
        ----------
        function_chain : List[Callable], optional
            List of functions to be executed in the chain, by default None.
        """

        self.functions = function_chain if (function_chain is not None) else []

        self.validate_chain()

    def __call__(self, **kwargs):
        """
        Executes all the functions in the chain.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        Any
            The result of the last function in the chain.

        Raises
        ------
        Exception
            If the execution fails at any function in the chain.
        """

        local_state = {}
        local_state.update(kwargs)
        try:
            returned_state = None
            for function in self.functions:
                signature = inspect.signature(function)
                func_kwargs = {key: value for key, value in local_state.items() if key in signature.parameters}
                returned_state = function(**func_kwargs)
                local_state.update(returned_state)
            else:
                result = returned_state
        except Exception as error:
            raise Exception(f"Execution failed processing function {function.__name__}. Error: {str(error)}")

        return result

    def add_function(self, function: typing.Callable, position: int = -1):
        """
        Adds a function to the chain at a specified position. Default is end of the chain.

        Parameters
        ----------
        function : Callable
            The function to add to the chain.
        position : int, optional
            The position in the chain to add the function to, by default -1
            which corresponds to the end of the chain.
        """

        if position < 0:
            position = len(self.functions)

        self.functions.insert(position, function)
        self.validate_chain()

    def remove_function(self, function: typing.Callable):
        """
        Removes a specific function from the chain.

        Parameters
        ----------
        function : Callable
            The function to remove from the chain.
        """

        self.functions.remove(function)
        self.validate_chain()

    def replace_function(self, old_function: typing.Callable, new_function: typing.Callable):
        """
        Replaces a specific function in the chain with a new one.

        Parameters
        ----------
        old_function : Callable
            The function to be replaced.
        new_function : Callable
            The function to replace the old one.
        """

        index = self.functions.index(old_function)
        self.functions[index] = new_function
        self.validate_chain()

    def validate_chain(self):
        """
        Validates the function chain by ensuring all objects in the chain are callable
        and all functions except the last one return a dictionary.

        Raises
        ------
        ValueError
            If an object in the chain is not callable or a function does not return a dictionary.
        """

        for idx, function in enumerate(self.functions):
            if not callable(function):
                raise ValueError(f"Object {function} is not callable.")

            signature = inspect.signature(function)

            if (idx < len(self.functions) - 1):
                if ((not signature.return_annotation is dict)
                        and (not typing.get_origin(signature.return_annotation) is dict)):
                    raise ValueError(f"Function {function.__name__} must return a dictionary. {signature}")
