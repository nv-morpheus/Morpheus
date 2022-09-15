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
import os
import time
import typing

import pandas as pd
import srf
from srf.core import operators as ops

import cudf

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.column_info import DataFrameInputSchema
from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPS3ToDataFrameStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 file_type: FileTypes,
                 input_schema: DataFrameInputSchema,
                 filter_null: bool = True,
                 cache_dir: str = "./.cache/dfp"):
        super().__init__(c)

        self._input_schema: DataFrameInputSchema = input_schema

        self._batch_size = 10
        self._batch_cache = []
        self._file_type = file_type
        self._filter_null = filter_null
        self._cache_dir = os.path.join(cache_dir, "s3_data")

    @property
    def name(self) -> str:
        return "dfp-s3-to-df"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        # boto3.resources.factory.s3.ObjectSummary?
        return (typing.Any, )

    def single_object_to_dataframe(self, s3_object):

        if (hasattr(s3_object, "DEBUG_JSON")):
            return pd.read_json(getattr(s3_object, "DEBUG_JSON"))
        else:
            cache_location = os.path.join(self._cache_dir, "raw", s3_object.bucket_name, s3_object.key + ".pkl")

            s3_filename = f"s3://{s3_object.bucket_name}/{s3_object.key}"

            if (not os.path.exists(cache_location)):

                # Make the directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_location), exist_ok=True)

                logger.debug("Downloading S3 object: %s", s3_filename)

                s3_df = read_file_to_df(s3_filename,
                                        self._file_type,
                                        filter_nulls=self._filter_null,
                                        df_type="pandas",
                                        parser_kwargs={
                                            "lines": False, "orient": "records"
                                        })

                # Do some simple pre-processing to filter the data as much as possible
                s3_df = process_dataframe(s3_df, self._input_schema)

                # Write the file cache
                s3_df.to_pickle(cache_location)

                # # Also write the CSV for easy reading
                # cache_location_csv = os.path.join(os.path.curdir,
                #                                   "s3_cache",
                #                                   s3_object.bucket_name,
                #                                   os.path.splitext(s3_object.key)[0] + ".csv")

                # s3_df.to_csv(cache_location_csv)
            else:
                try:
                    s3_df = pd.read_pickle(cache_location)
                except Exception as e:
                    raise

            return s3_df

    def _get_or_create_dataframe_from_s3_batch(
            self, s3_object_batch: typing.Union[typing.Any,
                                                typing.List[typing.Any]]) -> typing.Tuple[cudf.DataFrame, bool]:
        if (not s3_object_batch):
            return None, False

        if (isinstance(s3_object_batch, list)):

            # [(object, batch_count), ...]
            bucket_name = s3_object_batch[0][0].bucket_name
            batch_count = s3_object_batch[0][1]

            assert all([s3_object[0].bucket_name == bucket_name for s3_object in
                        s3_object_batch]), "Batches must come from a single bucket"

            # Create a list of dictionaries that only contains the information we are interested in hashing
            hash_data = [{
                "bucket_name": s3_object[0].bucket_name,
                "e_tag": s3_object[0].e_tag,
                "key": s3_object[0].key,
                "last_modified": s3_object[0].last_modified.strftime('%Y-%m-%d %H:%M:%S.%f'),
            } for s3_object in s3_object_batch]

            # Convert to base 64 encoding to remove - values
            # objects_hash_hex = hex(objects_hash & (2**64 - 1))
            objects_hash_hex = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

            batch_cache_location = os.path.join(self._cache_dir, "batches", bucket_name, f"{objects_hash_hex}.pkl")

            # Return the cache if it exists
            if (os.path.exists(batch_cache_location)):
                output_df = pd.read_pickle(batch_cache_location)
                output_df["origin_hash"] = objects_hash_hex
                output_df["batch_count"] = batch_count

                return (output_df, True)

            # Loop over dataframes and concat into one
            dfs = [self.single_object_to_dataframe(s3_object[0]) for s3_object in s3_object_batch]

            output_df: pd.DataFrame = pd.concat(dfs)

            # After concat, we need to sort by time and reset the index
            output_df.sort_values(self._config.ae.timestamp_column_name, ignore_index=True, inplace=True)

            # Save dataframe to cache future runs
            os.makedirs(os.path.dirname(batch_cache_location), exist_ok=True)

            try:
                output_df.to_pickle(batch_cache_location)
            except:
                logger.warning("Failed to save batch cache. Skipping cache for this batch.", exc_info=True)

            output_df["batch_count"] = batch_count
            output_df["origin_hash"] = objects_hash_hex

            return (output_df, False)
        else:
            output_df = self.single_object_to_dataframe(s3_object_batch)
            output_df["batch_count"] = 1
            output_df["origin_hash"] = None  # TODO(Devin)

            return (output_df, False)

    def convert_to_dataframe(self, s3_object_batch: typing.Union[typing.Any, typing.List[typing.Any]]):
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
            obs.pipe(ops.map(self.convert_to_dataframe)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, cudf.DataFrame
