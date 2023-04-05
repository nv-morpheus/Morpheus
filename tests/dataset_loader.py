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
import typing

import pandas as pd

import cudf

from morpheus.io.deserializers import read_file_to_df
from utils import TEST_DIRS


class DatasetLoader:
    """
    Helper class for loading and caching test datasets as DataFrames.

    Parameters
    ----------
    df_type : typing.Literal['cudf', 'pandas']
        Type of DataFrame to return unless otherwise explicitly specified.
    """

    __df_cache = {}  # {(df_type, path): DF}

    def __init__(self, df_type: typing.Literal['cudf', 'pandas']) -> None:
        self._default_df_type = df_type

    def get_alt_df_type(self, df_type: typing.Literal['cudf', 'pandas']) -> typing.Literal['cudf', 'pandas']:
        """Returns the other possible df type."""
        return 'cudf' if df_type == 'pandas' else 'pandas'

    def clear(self):
        self.__df_cache.clear()

    def get_df(self,
               file_path: str,
               df_type: typing.Literal['cudf', 'pandas'] = None) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
        """
        Fetch a DataFrame specified from `file_path`. If `file_path` is not an absolute path, it is assumed to be
        relative to the `test/tests_data` dir. If a DataFrame matching both the path and `df_type` has already been fetched, then a cached copy will be
        returned.
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
                    df = cudf.DataFrame.from_pandas(alt_df)
            else:
                df = read_file_to_df(full_path, df_type=df_type)

            self.__df_cache[(df_type, full_path)] = df

        return df.copy(deep=True)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )

        return self.get_df(*item)

    def get_loader(self, df_type: typing.Literal['cudf', 'pandas']):
        """
        Factory method to return an instance of `DatasetLoader` for the given df_type, returns `self` if the df_type
        matches. Used by cudf and pandas propery methods.
        """
        if self._default_df_type == df_type:
            return self
        else:
            return DatasetLoader(df_type=df_type)

    @property
    def cudf(self):
        return self.get_loader(df_type='cudf')

    @property
    def pandas(self):
        return self.get_loader(df_type='pandas')

    @staticmethod
    def repeat(df, repeat_count=2, reset_index=True) -> pd.DataFrame:
        """
        Returns a DF consisting of `repeat_count` copies of the original
        """
        if isinstance(df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cudf.concat

        repeated_df = concat_fn([df for _ in range(repeat_count)])

        if reset_index:
            repeated_df = repeated_df.reset_index(inplace=False, drop=True)

        return repeated_df
