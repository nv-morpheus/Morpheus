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

import json
import logging
import typing

import pandas as pd

from morpheus.cli import register_stage
from morpheus.config import PipelineModes
from morpheus.stages.input.autoencoder_source_stage import AutoencoderSourceStage

logger = logging.getLogger(__name__)


@register_stage("from-duo", modes=[PipelineModes.AE])
class DuoSourceStage(AutoencoderSourceStage):

    @property
    def name(self) -> str:
        return "from-duo"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def change_columns(df):
        df.columns = df.columns.str.replace('[_,.,{,},:]', '')
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: typing.List[str]):

        _DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'
        timestamp_column = "isotimestamp"
        city_column = "accessdevicelocationcity"
        state_column = "accessdevicelocationstate"
        country_column = "accessdevicelocationcountry"

        df['time'] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df['day'] = df['time'].dt.date
        df.fillna({'time': pd.to_datetime(_DEFAULT_DATE), 'day': pd.to_datetime(_DEFAULT_DATE).date()}, inplace=True)
        df.sort_values(by=['time'], inplace=True)

        overall_location_columns = [col for col in [city_column, state_column, country_column] if col is not None]
        overall_location_df = df[overall_location_columns].fillna('nan')
        df['overall_location'] = overall_location_df.apply(lambda x: ', '.join(x), axis=1)
        df['loc_cat'] = df.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
        df.fillna({'loc_cat': 1}, inplace=True)
        df['locincrement'] = df.groupby('day')['loc_cat'].expanding(1).max().droplevel(0)
        df.drop(['overall_location', 'loc_cat'], inplace=True, axis=1)

        df["logcount"] = df.groupby('day').cumcount()

        if (feature_columns is not None):
            df.drop(columns=df.columns.difference(feature_columns), inplace=True)

        return df

    @staticmethod
    def files_to_dfs_per_user(x: typing.List[str],
                              userid_column_name: str,
                              feature_columns: typing.List[str],
                              userid_filter: str = None,
                              repeat_count: int = 1) -> typing.Dict[str, pd.DataFrame]:

        dfs = []
        for file in x:
            with open(file) as json_in:
                log = json.load(json_in)
            df = pd.json_normalize(log)
            df = DuoSourceStage.change_columns(df)
            dfs = dfs + AutoencoderSourceStage.repeat_df(df, repeat_count)

        df_per_user = AutoencoderSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
