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

import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
import typing
from functools import partial

import fsspec
import pandas as pd
import srf
from srf.core import operators as ops

import dask
from dask.distributed import Client
from dask.distributed import LocalCluster

import cudf

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.column_info import DataFrameInputSchema
from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


def _single_object_to_dataframe(file_object: fsspec.core.OpenFile,
                                schema: DataFrameInputSchema,
                                file_type: FileTypes,
                                filter_null: bool,
                                parser_kwargs: dict):

    retries = 0
    s3_df = None
    while (retries < 2):
        try:
            with file_object as f:
                s3_df = read_file_to_df(f,
                                        file_type,
                                        filter_nulls=filter_null,
                                        df_type="pandas",
                                        parser_kwargs=parser_kwargs)

            break
        except Exception as e:
            if (retries < 2):
                logger.warning("Refreshing S3 credentials")
                # cred_refresh()
                retries += 1
            else:
                raise e

    # Run the pre-processing before returning
    if (s3_df is None):
        return s3_df

    s3_df = process_dataframe(df_in=s3_df, input_schema=schema)

    return s3_df


class DFPFileToDataFrameStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 schema: DataFrameInputSchema,
                 filter_null: bool = True,
                 file_type: FileTypes = FileTypes.Auto,
                 parser_kwargs: dict = None,
                 cache_dir: str = "./.cache/dfp"):
        super().__init__(c)

        self._schema = schema

        self._file_type = file_type
        self._filter_null = filter_null
        self._parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        self._cache_dir = os.path.join(cache_dir, "file_cache")

        self._dask_cluster: Client = None

        self._download_method: typing.Literal["single_thread", "multiprocess", "dask",
                                              "dask_thread"] = os.environ.get("MORPHEUS_FILE_DOWNLOAD_TYPE",
                                                                              "dask_thread")

    @property
    def name(self) -> str:
        return "dfp-s3-to-df"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def _get_dask_cluster(self):

        if (self._dask_cluster is None):
            logger.debug("Creating dask cluster...")

            # Up the heartbeat interval which can get violated with long download times
            dask.config.set({"distributed.client.heartbeat": "30s"})

            self._dask_cluster = LocalCluster(start=True, processes=not self._download_method == "dask_thread")

            logger.debug("Creating dask cluster... Done. Dashboard: %s", self._dask_cluster.dashboard_link)

        return self._dask_cluster

    def _close_dask_cluster(self):
        if (self._dask_cluster is not None):
            logger.debug("Stopping dask cluster...")

            self._dask_cluster.close()

            self._dask_cluster = None

            logger.debug("Stopping dask cluster... Done.")

    def _get_or_create_dataframe_from_s3_batch(
            self, file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> typing.Tuple[cudf.DataFrame, bool]:

        if (not file_object_batch):
            return None, False

        file_list = file_object_batch[0]
        batch_count = file_object_batch[1]

        fs: fsspec.AbstractFileSystem = file_list.fs

        # Create a list of dictionaries that only contains the information we are interested in hashing. `ukey` just
        # hashes all of the output of `info()` which is perfect
        hash_data = [{"ukey": fs.ukey(file_object.path)} for file_object in file_list]

        # Convert to base 64 encoding to remove - values
        objects_hash_hex = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

        batch_cache_location = os.path.join(self._cache_dir, "batches", f"{objects_hash_hex}.pkl")

        # Return the cache if it exists
        if (os.path.exists(batch_cache_location)):
            output_df = pd.read_pickle(batch_cache_location)
            output_df["origin_hash"] = objects_hash_hex
            output_df["batch_count"] = batch_count

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
            dfs = []
            if (self._download_method.startswith("dask")):

                # Create the client each time to ensure all connections to the cluster are closed (they can time out)
                with Client(self._get_dask_cluster()) as client:
                    dfs = client.map(download_method, download_buckets)

                    dfs = client.gather(dfs)

            elif (self._download_method == "multiprocessing"):
                # Use multiprocessing here since parallel downloads are a pain
                with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
                    dfs = p.map(download_method, download_buckets)
            else:
                # Simply loop
                for s3_object in download_buckets:
                    dfs.append(download_method(s3_object))

        except Exception as e:
            logger.exception("Failed to download logs. Error: ", exc_info=True)
            return None, False

        if (not dfs):
            logger.error("No logs were downloaded")
            return None, False

        output_df: pd.DataFrame = pd.concat(dfs)

        # Finally sort by timestamp and then reset the index
        output_df.sort_values(by=["timestamp"], inplace=True)

        output_df.reset_index(drop=True, inplace=True)

        # Save dataframe to cache future runs
        os.makedirs(os.path.dirname(batch_cache_location), exist_ok=True)

        try:
            output_df.to_pickle(batch_cache_location)
        except:
            logger.warning("Failed to save batch cache. Skipping cache for this batch.", exc_info=True)

        output_df["batch_count"] = batch_count
        output_df["origin_hash"] = objects_hash_hex

        return (output_df, False)

    def convert_to_dataframe(self, s3_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]):
        if (not s3_object_batch):
            return None

        start_time = time.time()

        try:

            output_df, cache_hit = self._get_or_create_dataframe_from_s3_batch(s3_object_batch)

            duration = (time.time() - start_time) * 1000.0

            logger.debug("S3 objects to DF complete. Rows: %s, Cache: %s, Duration: %s ms",
                         len(output_df),
                         "hit" if cache_hit else "miss",
                         duration)

            return output_df
        except Exception as e:
            logger.exception("Error while converting S3 buckets to DF.")
            self._get_or_create_dataframe_from_s3_batch(s3_object_batch)
            raise

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.convert_to_dataframe), ops.on_completed(self._close_dask_cluster)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, cudf.DataFrame
