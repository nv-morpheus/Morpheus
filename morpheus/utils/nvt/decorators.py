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


import cudf
import inspect
import typing
import pandas as pd


def sync_df_as_pandas(df_arg_name='df'):
    """
    This function serves as a decorator that synchronizes cudf.DataFrame to pandas.DataFrame before applying the function.

    Parameters
    ----------
    df_arg_name : str
        The name of the DataFrame parameter in the decorated function.

    Returns
    -------
    Callable
        The decorator.
    """

    def decorator(func: typing.Callable[..., pd.DataFrame]) -> typing.Callable[
        ..., typing.Union[pd.DataFrame, cudf.DataFrame]]:
        """
        The actual decorator that wraps the function.

        Parameters
        ----------
        func : Callable
            The function to apply to the DataFrame.

        Returns
        -------
        Callable
            The wrapped function.
        """

        def wrapper(*args, **kwargs) -> typing.Union[pd.DataFrame, cudf.DataFrame]:
            is_arg = False
            arg_index = 0
            df_arg = kwargs.get(df_arg_name)
            if df_arg is None:
                # try to get DataFrame argument from positional arguments
                func_args = inspect.signature(func).parameters
                for i, arg in enumerate(func_args):
                    if arg == df_arg_name:
                        is_arg = True
                        arg_index = i
                        df_arg = args[i]
                        break

            convert_to_cudf = False
            if type(df_arg) == cudf.DataFrame:
                convert_to_cudf = True
                if (is_arg):
                    args = list(args)
                    args[arg_index] = df_arg.to_pandas()
                    args = tuple(args)
                else:
                    kwargs[df_arg_name] = df_arg.to_pandas()

            result = func(*args, **kwargs)

            if convert_to_cudf:
                result = cudf.from_pandas(result)

            return result

        return wrapper

    return decorator
