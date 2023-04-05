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

import os
import random
import typing

import cupy as cp
import pandas as pd

import cudf as cdf  # rename to avoid clash with property method

from morpheus.io.deserializers import read_file_to_df
from utils import TEST_DIRS


class DatasetLoader:
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

    def __init__(self, df_type: typing.Literal['cudf', 'pandas']) -> None:
        self._default_df_type = df_type
        self.__instances[df_type] = self

    def get_alt_df_type(self, df_type: typing.Literal['cudf', 'pandas']) -> typing.Literal['cudf', 'pandas']:
        """Returns the other possible df type."""
        return 'cudf' if df_type == 'pandas' else 'pandas'

    def clear(self):
        self.__df_cache.clear()

    def get_df(self,
               file_path: str,
               df_type: typing.Literal['cudf', 'pandas'] = None) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        """
        Fetch a DataFrame specified from `file_path`. If `file_path` is not an absolute path, it is assumed to be
        relative to the `test/tests_data` dir. If a DataFrame matching both `file_path` and `df_type` has already been
        fetched, then a cached copy will be returned. In the event that a DataFrame matching `file_path` but not
        `df_type` exists in the cache, then the cached copy will be cast to the appropriate type, stored in the cache
        and then returned
        """
        if os.path.abspath(file_path) != os.path.normpath(file_path):
            full_path = os.path.join(TEST_DIRS.tests_data_dir, file_path)
        else:
            full_path = file_path

        if df_type is None:
            df_type = self._default_df_type

        df = self.__df_cache.get((df_type, full_path))
        if df is None:
            # If it isn't in the cache, but we have a cached copy in another DF format use it instead of going to disk
            alt_df_type = self.get_alt_df_type(df_type=df_type)
            alt_df = self.__df_cache.get((alt_df_type, full_path))
            if alt_df is not None:
                if alt_df_type == 'cudf':
                    df = alt_df.to_pandas()
                else:
                    df = cdf.DataFrame.from_pandas(alt_df)
            else:
                df = read_file_to_df(full_path, df_type=df_type)

            self.__df_cache[(df_type, full_path)] = df

        return df.copy(deep=True)

    def __getitem__(
        self, item: typing.Union[str, typing.Tuple[str], typing.Tuple[str, typing.Literal['cudf', 'pandas']]]
    ) -> typing.Union[cdf.DataFrame, pd.DataFrame]:
        if not isinstance(item, tuple):
            item = (item, )

        return self.get_df(*item)

    @classmethod
    def get_loader(cls, df_type: typing.Literal['cudf', 'pandas']):
        """
        Factory method to return an instance of `DatasetLoader` for the given df_type, returns `self` if the df_type
        matches. Used by cudf and pandas propery methods.
        """
        try:
            loader = cls.__instances[df_type]
        except KeyError:
            loader = cls(df_type=df_type)

        return loader

    @property
    def cudf(self):
        return self.get_loader(df_type='cudf')

    @property
    def pandas(self):
        return self.get_loader(df_type='pandas')

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
