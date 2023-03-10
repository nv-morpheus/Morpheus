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

import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import time
import typing
from functools import partial

import fsspec
import fsspec.utils
import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus._lib.common import FileTypes
from morpheus.cli.utils import str_to_file_type
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILE_TO_DF, MORPHEUS_MODULE_NAMESPACE)
def file_to_df(builder: mrc.Builder):
    """
    This module reads data from the batched files into a dataframe after receiving input from the "FileBatcher" module.
    In addition to loading data from the disk, it has ability to load the file content from S3 buckets.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    ----------
    Configurable parameters:
        - cache_dir: Directory to cache the rolling window data
        - file_type: Type of the input file
        - filter_null: Whether to filter out null values
        - parser_kwargs: Keyword arguments to pass to the parser
        - schema: Schema of the input data
        - timestamp_column_name: Name of the timestamp column
    """

    config = builder.get_current_module_config()

    timestamp_column_name = config.get("timestamp_column_name", "timestamp")

    if ("schema" not in config) or (config["schema"] is None):
        raise ValueError("Input schema is required.")

    schema_config = config["schema"]
    schema_str = schema_config["schema_str"]
    encoding = schema_config["encoding"]

    file_type = config.get("file_type", "JSON")
    filter_null = config.get("filter_null", False)
    parser_kwargs = config.get("parser_kwargs", None)
    cache_dir = config.get("cache_dir", None)

    download_method: typing.Literal["single_thread", "multiprocess", "dask",
    "dask_thread"] = os.environ.get("MORPHEUS_FILE_DOWNLOAD_TYPE", "multiprocess")

    if (cache_dir is None):
        cache_dir = "./.cache"
        logger.warning("Cache directory not set. Defaulting to ./.cache")

    cache_dir = os.path.join(cache_dir, "file_cache")

    # Load input schema
    schema = pickle.loads(bytes(schema_str, encoding))

    try:
        file_type = str_to_file_type(file_type.lower())
    except Exception:
        raise ValueError("Invalid input file type '{}'. Available file types are: CSV, JSON".format(file_type))

    def get_dask_cluster():

        try:
            import dask
            from dask.distributed import LocalCluster
        except ModuleNotFoundError:
            raise Exception("Install 'dask' and 'distributed' to allow file downloads using dask mode.")

        logger.debug("Creating dask cluster...")

        # Up the heartbeat interval which can get violated with long download times
        dask.config.set({"distributed.client.heartbeat": "30s"})

        dask_cluster = LocalCluster(start=True, processes=not download_method == "dask_thread")

        logger.debug("Creating dask cluster... Done. Dashboard: %s", dask_cluster.dashboard_link)

        return dask_cluster

    def get_dask_client(dask_cluster):

        from dask.distributed import Client

        logger.debug("Creating dask client...")
        dask_client = Client(dask_cluster)
        logger.debug("Creating dask client %s ... Done.", dask_client)

        return dask_client

    def close_dask_cluster():
        if (dask_cluster is not None):
            logger.debug("Stopping dask cluster...")
            dask_cluster.close()

            logger.debug("Stopping dask cluster... Done.")

    def single_object_to_dataframe(file_object: fsspec.core.OpenFile,
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
                    retries += 1
                else:
                    raise e

        # Run the pre-processing before returning
        if (s3_df is None):
            return s3_df

        s3_df = process_dataframe(df_in=s3_df, input_schema=schema)

        return s3_df

    def get_or_create_dataframe_from_s3_batch(
            file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> typing.Tuple[cudf.DataFrame, bool]:

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

        batch_cache_location = os.path.join(cache_dir, "batches", f"{objects_hash_hex}.pkl")

        # Return the cache if it exists
        if (os.path.exists(batch_cache_location)):
            output_df = pd.read_pickle(batch_cache_location)
            output_df["origin_hash"] = objects_hash_hex
            output_df["batch_count"] = batch_count

            return (output_df, True)

        # Cache miss
        download_method_func = partial(single_object_to_dataframe,
                                       file_type=file_type,
                                       filter_null=filter_null,
                                       parser_kwargs=parser_kwargs)

        download_buckets = file_list

        # Loop over dataframes and concat into one
        try:
            dfs = []
            if (download_method.startswith("dask")):
                # Create the client each time to ensure all connections to the cluster are
                # closed (they can time out)
                with get_dask_client(dask_cluster) as client:
                    dfs = client.map(download_method_func, download_buckets)

                    dfs = client.gather(dfs)

            elif (download_method == "multiprocessing"):
                # Use multiprocessing here since parallel downloads are a pain
                with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
                    dfs = p.map(download_method_func, download_buckets)
            else:
                # Simply loop
                for s3_object in download_buckets:
                    dfs.append(download_method_func(s3_object))

        except Exception:
            logger.exception("Failed to download logs. Error: ", exc_info=True)
            return None, False

        if (not dfs):
            logger.error("No logs were downloaded")
            return None, False

        output_df: pd.DataFrame = pd.concat(dfs)

        # Finally sort by timestamp and then reset the index
        output_df.sort_values(by=[timestamp_column_name], inplace=True)

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

    def convert_to_dataframe(file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]):
        if (not file_object_batch):
            return None

        start_time = time.time()

        try:
            output_df, cache_hit = get_or_create_dataframe_from_s3_batch(file_object_batch)

            duration = (time.time() - start_time) * 1000.0

            logger.debug("S3 objects to DF complete. Rows: %s, Cache: %s, Duration: %s ms",
                         len(output_df),
                         "hit" if cache_hit else "miss",
                         duration)
            return output_df
        except Exception:
            logger.exception("Error while converting S3 buckets to DF.")
            raise

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(convert_to_dataframe), ops.on_completed(close_dask_cluster)).subscribe(sub)

    if (download_method.startswith("dask")):
        dask_cluster = get_dask_cluster()

    node = builder.make_node_full(FILE_TO_DF, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
