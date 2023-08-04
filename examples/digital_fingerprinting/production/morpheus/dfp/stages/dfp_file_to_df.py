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
"""Stage for converting fsspec file objects to a DataFrame."""

import hashlib
import json
import logging
import os
import time
import typing
from functools import partial

import fsspec
import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.downloader import Downloader

logger = logging.getLogger(f"morpheus.{__name__}")


def _single_object_to_dataframe(file_object: fsspec.core.OpenFile,
                                schema: DataFrameInputSchema,
                                file_type: FileTypes,
                                filter_null: bool,
                                parser_kwargs: dict) -> pd.DataFrame:
    retries = 0
    df = None
    while (retries < 2):
        try:
            with file_object as f:
                df = read_file_to_df(f,
                                     file_type,
                                     filter_nulls=filter_null,
                                     df_type="pandas",
                                     parser_kwargs=parser_kwargs)

            break
        except Exception as e:
            if (retries < 2):
                logger.warning("Error fetching %s: %s\nRetrying...", file_object, e)
                retries += 1

    # Optimistaclly prep the dataframe (Not necessary since this will happen again in process_dataframe, but it
    # increases performance significantly)
    if (schema.prep_dataframe is not None):
        df = schema.prep_dataframe(df)

    return df


class DFPFileToDataFrameStage(PreallocatorMixin, SinglePortStage):
    """
    Stage for converting fsspec file objects to a DataFrame, pre-processing the DataFrame according to `schema`, and
    caching fetched file objects. The file objects are fetched in parallel using `morpheus.utils.downloader.Downloader`,
    which supports multiple download methods indicated by the `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable.

    Refer to `morpheus.utils.downloader.Downloader` for more information on the supported download methods.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    schema : `morpheus.utils.column_info.DataFrameInputSchema`
        Input schema for the DataFrame.
    filter_null : bool, optional
        Whether to filter null values from the DataFrame.
    file_type : `morpheus.common.FileTypes`, optional
        File type of the input files. If `FileTypes.Auto`, the file type will be inferred from the file extension.
    parser_kwargs : dict, optional
        Keyword arguments to pass to the DataFrame parser.
    cache_dir : str, optional
        Directory to use for caching.
    """

    def __init__(self,
                 config: Config,
                 schema: DataFrameInputSchema,
                 filter_null: bool = True,
                 file_type: FileTypes = FileTypes.Auto,
                 parser_kwargs: dict = None,
                 cache_dir: str = "./.cache/dfp"):
        super().__init__(config)

        self._schema = schema

        self._file_type = file_type
        self._filter_null = filter_null
        self._parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        self._cache_dir = os.path.join(cache_dir, "file_cache")

        self._downloader = Downloader()

    @property
    def name(self) -> str:
        """Stage name."""
        return "dfp-file-to-df"

    def supports_cpp_node(self):
        """Whether this stage supports a C++ node."""
        return False

    def accepted_types(self) -> typing.Tuple:
        """Accepted input types."""
        return (typing.Any, )

    def _get_or_create_dataframe_from_batch(
            self, file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> typing.Tuple[pd.DataFrame, bool]:

        if (not file_object_batch):
            raise RuntimeError("No file objects to process")

        file_list = file_object_batch[0]
        batch_count = file_object_batch[1]

        file_system: fsspec.AbstractFileSystem = file_list.fs

        # Create a list of dictionaries that only contains the information we are interested in hashing. `ukey` just
        # hashes all the output of `info()` which is perfect
        hash_data = [{"ukey": file_system.ukey(file_object.path)} for file_object in file_list]

        # Convert to base 64 encoding to remove - values
        objects_hash_hex = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

        batch_cache_location = os.path.join(self._cache_dir, "batches", f"{objects_hash_hex}.pkl")

        # Return the cache if it exists
        if (os.path.exists(batch_cache_location)):
            output_df = pd.read_pickle(batch_cache_location)
            output_df["batch_count"] = batch_count
            output_df["origin_hash"] = objects_hash_hex

            return (output_df, True)

        # Cache miss
        download_method = partial(_single_object_to_dataframe,
                                  schema=self._schema,
                                  file_type=self._file_type,
                                  filter_null=self._filter_null,
                                  parser_kwargs=self._parser_kwargs)

        download_buckets = file_list

        # Loop over dataframes and concat into one
        try:
            dfs = self._downloader.download(download_buckets, download_method)
        except Exception:
            logger.exception("Failed to download logs. Error: ", exc_info=True)
            raise

        if (dfs is None or len(dfs) == 0):
            raise ValueError("No logs were downloaded")

        output_df: pd.DataFrame = pd.concat(dfs)
        output_df = process_dataframe(df_in=output_df, input_schema=self._schema)

        # Finally sort by timestamp and then reset the index
        output_df.sort_values(by=[self._config.ae.timestamp_column_name], inplace=True)

        output_df.reset_index(drop=True, inplace=True)

        # Save dataframe to cache future runs
        os.makedirs(os.path.dirname(batch_cache_location), exist_ok=True)

        try:
            output_df.to_pickle(batch_cache_location)
        except Exception:
            logger.warning("Failed to save batch cache. Skipping cache for this batch.", exc_info=True)

        output_df["batch_count"] = batch_count
        output_df["origin_hash"] = objects_hash_hex

        return (output_df, False)

    def convert_to_dataframe(self, fsspec_batch: typing.Tuple[fsspec.core.OpenFiles, int]):
        """Converts a batch of fsspec objects to a DataFrame."""
        if (not fsspec_batch):
            return None

        start_time = time.time()

        try:

            output_df, cache_hit = self._get_or_create_dataframe_from_batch(fsspec_batch)

            duration = (time.time() - start_time) * 1000.0

            if (output_df is not None and logger.isEnabledFor(logging.DEBUG)):
                logger.debug("fsspec objects to DF complete. Rows: %s, Cache: %s, Duration: %s ms, Rate: %s rows/s",
                             len(output_df),
                             "hit" if cache_hit else "miss",
                             duration,
                             len(output_df) / (duration / 1000.0))

            return output_df
        except Exception:
            logger.exception("Error while converting fsspec batch to DF.")
            raise

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = builder.make_node(self.unique_name,
                                   ops.map(self.convert_to_dataframe),
                                   ops.on_completed(self._downloader.close))
        builder.make_edge(input_stream[0], stream)

        return stream, pd.DataFrame
