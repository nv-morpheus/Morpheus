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

import io
import json
import logging
import re
import typing
from functools import partial
from json.decoder import JSONDecodeError

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.pipeline import SingleOutputSource
from morpheus.pipeline import StreamPair
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.utils.directory_watcher import DirectoryWatcher

logger = logging.getLogger(__name__)


@register_stage("from-appshield", modes=[PipelineModes.FIL])
class AppShieldSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage is used to load Appshield messages from one or more plugins into a dataframe.
    It normalizes nested json messages and arranges them into a dataframe by snapshot
    and source.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, `./input_dir/<source>/snapshot-*/*.json` would read all
        files with the 'json' extension in the directory input_dir.
    plugins_include : List[str], default = None
        Plugins for appshield to be extracted.
    cols_include : List[str], default = None
        Raw features to extract from appshield plugins data.
    cols_exclude : List[str], default = ["SHA256"]
        Columns that aren't essential should be excluded.
    watch_directory : bool, default = False
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    max_files : int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    sort_glob : bool, default = False
        If true the list of files matching `input_glob` will be processed in sorted order.
    recursive : bool, default = True
        If true, events will be emitted for the files in subdirectories matching `input_glob`.
    queue_max_size : int, default = 128
        Maximum queue size to hold the file paths to be processed that match `input_glob`.
    batch_timeout : float, default = 5.0
        Timeout to retrieve batch messages from the queue.
    encoding : str, default = latin1
        Encoding to read a file.
    """

    def __init__(self,
                 c: Config,
                 input_glob: str,
                 plugins_include: typing.List[str],
                 cols_include: typing.List[str],
                 cols_exclude: typing.List[str] = ["SHA256"],
                 watch_directory: bool = False,
                 max_files: int = -1,
                 sort_glob: bool = False,
                 recursive: bool = True,
                 queue_max_size: int = 128,
                 batch_timeout: float = 5.0,
                 encoding: str = 'latin1'):

        SingleOutputSource.__init__(self, c)

        self._plugins_include = plugins_include
        self._cols_include = cols_include
        self._cols_exclude = cols_exclude
        self._encoding = encoding

        self._input_count = None

        self._watcher = DirectoryWatcher(input_glob=input_glob,
                                         watch_directory=watch_directory,
                                         max_files=max_files,
                                         sort_glob=sort_glob,
                                         recursive=recursive,
                                         queue_max_size=queue_max_size,
                                         batch_timeout=batch_timeout)

    @property
    def name(self) -> str:
        return "from-appshield"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self):
        return False

    @staticmethod
    def fill_interested_cols(plugin_df: pd.DataFrame, cols_include: typing.List[str]):
        """
        Fill missing interested plugin columns.

        Parameters
        ----------
        plugin_df : pandas.DataFrame
            Snapshot plugin dataframe
        cols_include : typing.List[str]
            Columns that needs to be included.

        Returns
        -------
        pandas.DataFrame
            The columns added dataframe.
        """
        cols_exists = plugin_df.columns
        for col in cols_include:
            if col not in cols_exists:
                plugin_df[col] = None
        plugin_df = plugin_df[cols_include]

        return plugin_df

    @staticmethod
    def read_file_to_df(file: io.TextIOWrapper, cols_exclude: typing.List[str]):
        """
        Read file content to dataframe.

        Parameters
        ----------
        file : `io.TextIOWrapper`
            Input file object
        cols_exclude : typing.List[str]
            Dropping columns from a dataframe.

        Returns
        -------
        pandas.DataFrame
            The columns added dataframe
        """
        data = json.load(file)
        titles = data["titles"]
        features_plugin = [col for col in titles if col not in cols_exclude]

        try:
            plugin_df = pd.DataFrame(columns=features_plugin, data=data["data"])
        except ValueError:
            logger.exception("Error while loading file content to datframe with 'cols_exclude' filter")
            logger.info("Attempting to populate the dataframe with all columns.")

            plugin_df = pd.DataFrame(columns=titles, data=data["data"])

            logger.info("Applying 'cols_exclude' filter on dataframe")

            plugin_df = plugin_df[features_plugin]

        return plugin_df

    @staticmethod
    def load_df(filepath: str, cols_exclude: typing.List[str], encoding: str) -> pd.DataFrame:
        """
        Reads a file into a dataframe.

        Parameters
        ----------
        filepath : str
            Path to a file.
        cols_exclude : typing.List[str]
            Columns that needs to exclude.
        encoding : str
            Encoding to read a file.

        Returns
        -------
        pandas.DataFrame
            The parsed dataframe.

        Raises
        ------
        JSONDecodeError
            If not able to decode the json file.
        """

        try:
            with open(filepath, encoding=encoding) as file:
                plugin_df = AppShieldSourceStage.read_file_to_df(file, cols_exclude)
        except JSONDecodeError as decode_error:
            logger.error('Unable to load %s to dataframe with %s encoding : %s', filepath, encoding, decode_error)

            encoding = encoding.lower()
            # To avoid retrying with utf-8, check if the given encoding is utf.
            if encoding.startswith('utf'):
                raise decode_error

            logger.info('Retrying... Attempting to load %s with utf-8 encoding', filepath)

            with open(filepath, encoding='utf-8') as file:
                plugin_df = AppShieldSourceStage.read_file_to_df(file, cols_exclude)

        return plugin_df

    @staticmethod
    def load_meta_cols(filepath_split: typing.List[str], plugin: str, plugin_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads meta columns to dataframe.

        Parameters
        ----------
        filepath_split : typing.List[str]
            Splits of file path.
        plugin : str
            Plugin name to which the data belongs to.

        Returns
        -------
        pandas.DataFrame
            The parsed dataframe.
        """

        if len(filepath_split) < 3:
            raise ValueError('Invalid filepath_split {}. Length should be greater than 2'.format(filepath_split))

        source = filepath_split[-3]

        snapshot_id = int(filepath_split[-2].split('-')[1])
        ts_re = re.search('[a-z]+_([0-9-_.]+).json', filepath_split[-1])

        if ts_re is None:
            raise ValueError('Invalid format for filepath_split {}'.format(filepath_split))

        timestamp = ts_re.group(1)

        plugin_df['snapshot_id'] = snapshot_id
        plugin_df['timestamp'] = timestamp
        plugin_df['source'] = source
        plugin_df['plugin'] = plugin

        return plugin_df

    @staticmethod
    def batch_source_split(x: typing.List[pd.DataFrame], source: str) -> typing.Dict[str, pd.DataFrame]:
        """
        Combines plugin dataframes from multiple snapshot and split dataframe per source.

        Parameters
        ----------
        x : typing.List[pd.DataFrame]
            Dataframes from multiple sources.
        source : str
            source column name to group it.

        Returns
        -------
        typing.Dict[str, pandas.DataFrame]
            Grouped dataframes by source.
        """

        combined_df = pd.concat(x)

        # Get the sources in this DF
        unique_sources = combined_df[source].unique()

        source_dfs = {}

        if len(unique_sources) > 1:
            for source_name in unique_sources:
                source_dfs[source_name] = combined_df[combined_df[source] == source_name]
        else:
            source_dfs[unique_sources[0]] = combined_df

        return source_dfs

    @staticmethod
    def files_to_dfs(x: typing.List[str],
                     cols_include: typing.List[str],
                     cols_exclude: typing.List[str],
                     plugins_include: typing.List[str],
                     encoding: str) -> pd.DataFrame:
        """
        Load plugin files into a dataframe, then segment the dataframe by source.

        Parameters
        ----------
        x : typing.List[str]
            List of file paths.
        cols_include : typing.List[str]
            Columns that needs to include.
        cols_exclude : typing.List[str]
            Columns that needs to exclude.
        encoding : str
            Encoding to read a file.

        Returns
        -------
        typing.Dict[str, pandas.DataFrame]
            Grouped dataframes by source.
        """
        # Using pandas to parse nested JSON until cuDF adds support
        # https://github.com/rapidsai/cudf/issues/8827
        plugin_dfs = []
        for filepath in x:
            try:
                filepath_split = filepath.split('/')
                plugin = filepath_split[-1].split('_')[0]

                if plugin in plugins_include:
                    plugin_df = AppShieldSourceStage.load_df(filepath, cols_exclude, encoding)
                    plugin_df = AppShieldSourceStage.fill_interested_cols(plugin_df, cols_include)
                    plugin_df = AppShieldSourceStage.load_meta_cols(filepath_split, plugin, plugin_df)
                    plugin_dfs.append(plugin_df)

            except JSONDecodeError as decode_error:
                logger.error('Unable to decode json file %s: %s', filepath, decode_error)

        df_per_source = AppShieldSourceStage.batch_source_split(plugin_dfs, source='source')

        return df_per_source

    @staticmethod
    def _build_metadata(x: typing.Dict[str, pd.DataFrame]):

        metas = []

        for source, df in x.items():

            # Now make a AppShieldMessageMeta with the source name
            meta = AppShieldMessageMeta(df, source)
            metas.append(meta)

        return metas

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        # The first source just produces filenames
        filename_source = self._watcher.build_node(self.unique_name, builder)

        out_type = typing.List[str]

        # Supposed to just return a source here
        return filename_source, out_type

    def _post_build_single(self, builder: mrc.Builder, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
            obs.pipe(
                # At this point, we have batches of filenames to process. Make a node for processing batches of
                # filenames into batches of dataframes
                ops.map(
                    partial(self.files_to_dfs,
                            cols_include=self._cols_include,
                            cols_exclude=self._cols_exclude,
                            plugins_include=self._plugins_include,
                            encoding=self._encoding)),
                ops.map(self._build_metadata),
                # Finally flatten to single meta
                ops.flatten()).subscribe(sub)

        post_node = builder.make_node_full(self.unique_name + "-post", node_fn)
        builder.make_edge(out_stream, post_node)

        out_stream = post_node
        out_type = AppShieldMessageMeta

        return super()._post_build_single(builder, (out_stream, out_type))
