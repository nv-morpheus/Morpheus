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
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("appshield-partitioner", modes=[PipelineModes.FIL])
class AppshieldPartitionerStage(MultiMessageStage):

    def __init__(self,
                 c: Config,
                 interested_plugins: typing.List[str],
                 cols_include: typing.List[str],
                 plugins_schema: typing.Dict[str, any]):
        super().__init__(c)

        self._interested_plugins = interested_plugins
        self._cols_include = cols_include
        self._plugins_schema = plugins_schema
        self._plugin_df_dict = {}

    @property
    def name(self) -> str:
        return "appshield-partitioner"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        return False

    def _sources_with_multi_snapshots(self):
        multiple_snapshots = {}
        for source, source_dict in self._plugin_df_dict.items():
            if len(source_dict) > 1:
                for scan_id in source_dict.keys():
                    multiple_snapshots.setdefault(source, []).append(scan_id)
        return multiple_snapshots

    def _hold_plugin_df(self, source, scan_id, plugin, plugin_df):
        if source not in self._plugin_df_dict:
            self._plugin_df_dict[source] = {}
        source = self._plugin_df_dict[source]

        if scan_id not in source:
            source[scan_id] = {}
        snapshot = source[scan_id]

        if plugin not in snapshot:
            snapshot[plugin] = plugin_df
        else:
            snapshot[plugin] = pd.concat([snapshot[plugin], plugin_df])

    def _df_per_source(self, sources_with_multi_snapshots):
        source_dfs = {}
        for source in sources_with_multi_snapshots.keys():
            plugin_dfs = []

            max_snapshot_id = max(sources_with_multi_snapshots[source])
            snapshot_ids = [x for x in sources_with_multi_snapshots[source] if x != max_snapshot_id]

            for scan_id in snapshot_ids:
                plugin_df_dict = self._plugin_df_dict[source][scan_id]

                for plugin_name in plugin_df_dict.keys():
                    plugin_df = AppshieldPartitionerStage.fill_interested_cols(plugin_df=plugin_df_dict[plugin_name],
                                                                               cols_include=self._cols_include)
                    # Add fully recieved snapshot to processing list.
                    plugin_dfs.append(plugin_df)

                # Removing from in memory as it is prepared for processing.
                del self._plugin_df_dict[source][scan_id]

            if plugin_dfs:
                source_dfs[source] = pd.concat(plugin_dfs, ignore_index=True)
        return source_dfs

    @staticmethod
    def fill_interested_cols(plugin_df: pd.DataFrame, cols_include: typing.List[str]):
        cols_exists = plugin_df.columns
        for col in cols_include:
            if col not in cols_exists:
                plugin_df[col] = None
        plugin_df = plugin_df[cols_include]

        return plugin_df

    def partition_by_source_snapshot(self, x: MessageMeta):

        df = x.df.to_pandas()

        unique_sources = df.source.unique()
        unique_scan_ids = df.scan_id.unique()

        for source in unique_sources:
            for scan_id in unique_scan_ids:
                for intrested_plugin in self._interested_plugins:
                    plugin_schema = self._plugins_schema[intrested_plugin]
                    plugin = plugin_schema.get("name")
                    plugin_df = df[(df.type_name == plugin) & (df.source == source) & (df.scan_id == scan_id)]

                    if not plugin_df.empty:
                        plugin_df = plugin_df.rename(columns=plugin_schema.get("column_mapping"))
                        plugin_df["plugin"] = intrested_plugin
                        self._hold_plugin_df(source, scan_id, intrested_plugin, plugin_df)

        sources_with_multi_snapshots = self._sources_with_multi_snapshots()

        source_dfs = self._df_per_source(sources_with_multi_snapshots)

        return source_dfs

    def build_metadata(self, x: typing.Dict[str, pd.DataFrame]):

        metas = []

        for source, df in x.items():
            # Now make a AppShieldMessageMeta with the source name
            meta = AppShieldMessageMeta(df, source)
            metas.append(meta)

        return metas

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        node = builder.make_node(self.unique_name, ops.map(self.partition_by_source_snapshot), ops.map(self.build_metadata), ops.flatten())
        builder.make_edge(stream, node)
        stream = node

        return stream, AppShieldMessageMeta

