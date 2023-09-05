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

import datetime
import typing

import mrc
from mrc.core import operators as ops

from dask.distributed import Client

from common.data_models import FeatureConfig
from common.feature_extractor import FeatureExtractor
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.appshield_source_stage import AppShieldMessageMeta


@register_stage("create-features", modes=[PipelineModes.FIL])
class CreateFeaturesRWStage(SinglePortStage):
    """
    This class extends MultiMessageStage to deal with scenario specific features from Appshiled plugins data.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    interested_plugins : typing.List[str]
        Only intrested plugins files will be read from Appshield snapshots
    feature_columns : typing.List[str]
        List of features needed to be extracted.
    file_extns : typing.List[str]
        File extensions.
    n_workers: int, default = 2
        Number of dask workers.
    threads_per_worker: int, default = 2
        Number of threads for each dask worker.
    """

    def __init__(
        self,
        c: Config,
        interested_plugins: typing.List[str],
        feature_columns: typing.List[str],
        file_extns: typing.List[str],
        n_workers: int = 2,
        threads_per_worker: int = 2,
    ):
        # self._client = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
        self._feature_config = FeatureConfig(file_extns, interested_plugins)
        self._feas_all_zeros = dict.fromkeys(feature_columns, 0)

        # FeatureExtractor instance to extract features from the snapshots.
        self._fe = FeatureExtractor(self._feature_config)

        super().__init__(c)

    @property
    def name(self) -> str:
        return "create-features-rw"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        """
        return (AppShieldMessageMeta, )

    def supports_cpp_node(self):
        return False


    def on_next(self, x: AppShieldMessageMeta):

        df = x.df

        df["Name"] = df["Name"].str.lower()

        # Create PID_Process feature.
        df['PID_Process'] = df.PID.astype(str) + '_'# + df.Process

        features_df = self._fe.extract_features(df, feas_all_zeros=self._feas_all_zeros)
        features_df['source_pid_process'] = x.source + '_' + features_df.pid_process
        features_df.index = features_df.source_pid_process

        # Create AppShieldMessageMeta with extracted features information.
        meta = AppShieldMessageMeta(features_df, x.source)

        return meta

    def on_completed():
        # Close dask client when pipeline initiates shutdown
        pass
        # self._client.close()

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        node = builder.make_node(self.unique_name, ops.map(self.on_next), ops.on_completed(self.on_completed))
        builder.make_edge(stream, node)
        stream = node

        return stream, AppShieldMessageMeta

