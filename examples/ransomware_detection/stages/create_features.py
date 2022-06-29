# Copyright (c) 2022, NVIDIA CORPORATION.
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

import typing

import srf
from common.data_models import FeatureConfig
from common.feature_extractor import FeatureExtractor
from srf.core import operators as ops

from dask.distributed import Client

from morpheus._lib.messages import MessageMeta
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.appshield_source_stage import AppShieldMessageMeta


class CreateFeaturesRWStage(MultiMessageStage):
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
        self._client = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
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

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        def node_fn(input: srf.Observable, output: srf.Subscriber):

            def on_next(x: AppShieldMessageMeta):

                snapshot_fea_dfs = []

                df = x.df

                # Type cast CommitCharge.
                df["CommitCharge"] = df["CommitCharge"].astype("float").astype("Int32")
                df["Name"] = df["Name"].str.lower()

                # Create PID_Process feature.
                df['PID_Process'] = df.PID + '_' + df.Process

                snapshot_ids = df.snapshot_id.unique()

                if len(snapshot_ids) > 1:
                    # Group snapshot rows using snapshot id.
                    all_dfs = [df[df.snapshot_id == snapshot_id] for snapshot_id in snapshot_ids]
                else:
                    all_dfs = [df]

                extract_func = self._fe.extract_features
                combine_func = FeatureExtractor.combine_features

                # Schedule dask task `extract_features` per snapshot.
                snapshot_fea_dfs = self._client.map(extract_func, all_dfs, feas_all_zeros=self._feas_all_zeros)

                # Combined `extract_features` results.
                features_df = self._client.submit(combine_func, snapshot_fea_dfs)

                # Gather features from all the snapshots.
                features_df = features_df.result()

                # Snapshot sequence will be generated using `source_pid_process`.
                # Determines which source generated the snapshot messages.
                # There's a chance of receiving the same snapshots names from multiple sources(hosts)
                features_df['source_pid_process'] = x.source + '_' + features_df.pid_process

                # Sort entries by pid_process and snapshot_id
                features_df = features_df.sort_values(by=["pid_process", "snapshot_id"]).reset_index(drop=True)

                # Create AppShieldMessageMeta with extracted features information.
                meta = AppShieldMessageMeta(features_df, x.source)

                return meta

            def create_multi_messages(x: MessageMeta) -> typing.List[MultiMessage]:

                multi_messages = []

                df = x.df

                pid_processes = df.pid_process.unique()

                # Create multi messaage per pid_process
                for pid_process in pid_processes:

                    pid_process_index = df[df.pid_process == pid_process].index

                    start = pid_process_index.min()
                    stop = pid_process_index.max() + 1
                    mess_count = stop - start

                    multi_message = MultiMessage(meta=x, mess_offset=start, mess_count=mess_count)
                    multi_messages.append(multi_message)

                return multi_messages

            def on_completed():
                # Close dask client when pipeline initiates shutdown
                self._client.close()

            input.pipe(ops.map(on_next), ops.map(create_multi_messages), ops.on_completed(on_completed),
                       ops.flatten()).subscribe(output)

        node = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(stream, node)
        stream = node

        return stream, MultiMessage
