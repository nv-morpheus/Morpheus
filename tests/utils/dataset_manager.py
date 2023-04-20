# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import typing

import cupy as cp
import pandas as pd

import cudf as cdf  # rename to avoid clash with property method

from morpheus.io.deserializers import read_file_to_df
from morpheus.utils import compare_df
from utils import TEST_DIRS


class DatasetManager(object):
    """
    Helper class for loading and caching test datasets as DataFrames, along with some common manipulation methods.

    Parameters
    ----------
    df_type : typing.Literal['cudf', 'pandas']
        Type of DataFrame to return unless otherwise explicitly specified.
    """

    __df_cache: typing.Dict[typing.Tuple[typing.Literal['cudf', 'pandas'], str],
                            typing.Union[cdf.DataFrame, pd.DataFrame]] = {}

    # Values in `__instances` are instances of `DatasetLoader`
    __instances: typing.Dict[typing.Literal['cudf', 'pandas'], typing.Any] = {}

    # Explicitly using __new__ instead of of an __init__ to implement this as a singleton for each dataframe type.
    # Initialization is also being performed here instead of an __init__ method as an __init__ method would be re-run
    # the __init__ on the singleton instance for each cache hit.
    def __new__(cls, df_type: typing.Literal['cudf', 'pandas']):
        try:
            return cls.__instances[df_type]
        except KeyError:
            instance = super().__new__(cls)
            instance._default_df_type = df_type
            cls.__instances[df_type] = instance
            return instance

    @staticmethod
    def get_alt_df_type(df_type: typing.Literal['cudf', 'pandas']) -> typing.Literal['cudf', 'pandas']:
        """Returns the other possible df type."""
        return 'cudf' if df_type == 'pandas' else 'pandas'

    def clear(self):
        self.__df_cache.clear()

    def get_df(self,
               file_path: str,
               df_type: typing.Literal['cudf', 'pandas'] = None,
               no_cache: bool = False,
               **reader_kwargs) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        """
        Fetch a DataFrame specified from `file_path`. If `file_path` is not an absolute path, it is assumed to be
        relative to the `test/tests_data` dir. If a DataFrame matching both `file_path` and `df_type` has already been
        fetched, then a cached copy will be returned. In the event that a DataFrame matching `file_path` but not
        `df_type` exists in the cache, then the cached copy will be cast to the appropriate type, stored in the cache
        and then returned

        Passing values to `reader_kwargs` will cause the cache to by bypassed
        """

        if len(reader_kwargs) and not no_cache:
            logger = logging.getLogger(f"morpheus.{__name__}")
            logger.warning("Setting specific `reader_kwargs` requires bypassing the cache. "
                           "Set `no_cache=True` to avoid this warning.")
            no_cache = True

        abs_path = os.path.abspath(file_path)
        if abs_path != os.path.normpath(file_path):
            # Relative paths are assumed to be relative to the `tests/tests_data` dir
            full_path = os.path.abspath(os.path.join(TEST_DIRS.tests_data_dir, file_path))
        else:
            full_path = abs_path

        if df_type is None:
            df_type = self.default_df_type

        df = None
        if not no_cache:
            df = self.__df_cache.get((df_type, full_path))

        if df is None:
            alt_df = None
            if not no_cache:
                # If it isn't in the cache, but we have a cached copy in another DF format use it instead of re-reading
                alt_df_type = self.get_alt_df_type(df_type=df_type)
                alt_df = self.__df_cache.get((alt_df_type, full_path))

            if alt_df is not None:
                if alt_df_type == 'cudf':
                    df = alt_df.to_pandas()
                else:
                    df = cdf.DataFrame.from_pandas(alt_df)
            else:
                df = read_file_to_df(full_path, df_type=df_type, **reader_kwargs)

            if not no_cache:
                self.__df_cache[(df_type, full_path)] = df

        return df.copy(deep=True)

    def __getitem__(
        self, item: typing.Union[str, typing.Tuple[str], typing.Tuple[str, typing.Literal['cudf', 'pandas']]]
    ) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        if not isinstance(item, tuple):
            item = (item, )

        return self.get_df(*item)

    @property
    def cudf(self):
        return DatasetManager(df_type='cudf')

    @property
    def pandas(self):
        return DatasetManager(df_type='pandas')

    @property
    def default_df_type(self):
        return self._default_df_type

    @staticmethod
    def repeat(df: typing.Union[cdf.DataFrame, pd.DataFrame],
               repeat_count: int = 2,
               reset_index: bool = True) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        """
        Returns a DF consisting of `repeat_count` copies of the original
        """
        if isinstance(df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cdf.concat

        repeated_df = concat_fn([df for _ in range(repeat_count)])

        if reset_index:
            repeated_df = repeated_df.reset_index(inplace=False, drop=True)

        return repeated_df

    @staticmethod
    def replace_index(df: typing.Union[cdf.DataFrame, pd.DataFrame],
                      replace_ids: typing.Dict[int, int]) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        """Return a new DataFrame's where we replace some index values with others."""
        return df.rename(index=replace_ids)

    @classmethod
    def dup_index(cls,
                  df: typing.Union[cdf.DataFrame, pd.DataFrame],
                  count: int = 1) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        """Randomly duplicate `count` entries in a DataFrame's index"""
        assert count * 2 <= len(df), "Count must be less than half the number of rows."

        # Sample 2x the count. One for the old ID and one for the new ID. Dont want duplicates so we use random.sample
        # (otherwise you could get less duplicates than requested if two IDs just swap)
        dup_ids = random.sample(df.index.values.tolist(), 2 * count)

        # Create a dictionary of old ID to new ID
        replace_dict = {x: y for x, y in zip(dup_ids[:count], dup_ids[count:])}

        # Return a new dataframe where we replace some index values with others
        return cls.replace_index(df, replace_dict)

    @staticmethod
    def assert_df_equal(df_to_check: typing.Union[pd.DataFrame, cdf.DataFrame], val_to_check: typing.Any) -> bool:
        """Compare a DataFrame against a validation dataset which can either be a DataFrame, Series or CuPy array."""

        # Comparisons work better in cudf so convert everything to that
        if (isinstance(df_to_check, cdf.DataFrame) or isinstance(df_to_check, cdf.Series)):
            df_to_check = df_to_check.to_pandas()

        if (isinstance(val_to_check, cdf.DataFrame) or isinstance(val_to_check, cdf.Series)):
            val_to_check = val_to_check.to_pandas()
        elif (isinstance(val_to_check, cp.ndarray)):
            val_to_check = val_to_check.get()

        bool_df = df_to_check == val_to_check

        return bool(bool_df.all(axis=None))

