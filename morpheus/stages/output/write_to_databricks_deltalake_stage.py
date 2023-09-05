# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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




import logging
import time
from io import StringIO
import os
import pandas as pd
import cudf
import typing
import dask

from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import LongType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import TimestampType

import mrc
from mrc.core import operators as ops
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.utils.databricks_utils import configure_databricks_connect

logger = logging.getLogger(__name__)


@register_stage("to-databricks-deltalake")
class DataBricksDeltaLakeSinkStage(SinglePortStage):
    """
    Sink stage used to write messages to a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    delta_path : str, default None
        Path of the delta table where the data need to be written or updated.
    databricks_host : str, default None
        URL of Databricks host to connect to.
    databricks_token : str, default None
        Access token for Databricks cluster.
    databricks_cluster_id : str, default None
        Databricks cluster to be used to query the data as per SQL provided.
    databricks_port : str, defailt "15001"
        Databricks port that Databricks Connect connects to. Defaults to 15001
    databricks_org_id : str, default "0"
        Azure-only, see ?o=orgId in URL. Defaults to 0 for other platform
    """

    def __init__(self,
                 config: Config,
                 delta_path: str = None,
                 databricks_host: str = None,
                 databricks_token: str = None,
                 databricks_cluster_id: str = None,
                 databricks_port: str = "15001",
                 databricks_org_id: str = "0"):

        super().__init__(config)
        self.delta_path = delta_path
        configure_databricks_connect(databricks_host,
                                           databricks_token,
                                           databricks_cluster_id,
                                           databricks_port,
                                           databricks_org_id)

        # Enable Arrow-based columnar data transfers
        self.spark = SparkSession.builder \
            .config("spark.databricks.delta.optimizeWrite.enabled", "true") \
            .config("spark.databricks.delta.autoCompact.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

    @property
    def name(self) -> str:
        return "to-databricks-deltalake"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.
        """
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):


            def write_to_deltalake(meta: MessageMeta):
            # convert cudf to spark dataframe
            df = meta.copy_dataframe()
            if isinstance(df, cudf.DataFrame):
                df = df.to_pandas()
            schema = self._extract_schema_from_pandas_dataframe(df)
            spark_df = self.spark.createDataFrame(df, schema=schema)
            spark_df.write \
                .format('delta') \
                .option("mergeSchema", "true") \
                .mode("append") \
                .save(self.delta_path)
            return meta
        node = builder.make_node(self.unique_name, ops.map(write_to_deltalake))
        builder.make_edge(stream, node)

        # Return input unchanged to allow passthrough
        return node, input_stream[1]

    @staticmethod
    def _extract_schema_from_pandas_dataframe(self, df: pd.DataFrame): -> StructType
        """
        Extract approximate schemas from pandas dataframe
        """
        spark_schema = []
        for col, dtype in df.dtypes.items():
            try:
                if dtype == "bool":
                    spark_dtype = StructField(col, BooleanType())
                elif dtype == "int64":
                    spark_dtype = StructField(col, LongType())
                elif dtype == "int32":
                    spark_dtype = StructField(col, IntegerType())
                elif dtype == "float64":
                    spark_dtype = StructField(col, DoubleType())
                elif dtype == "float32":
                    spark_dtype = StructField(col, FloatType())
                elif dtype == "datetime64[ns]":
                    spark_dtype = StructField(col, TimestampType())
                else:
                    spark_dtype = StructField(col, StringType())
            except Exception as e:
                logger.error(f"Encountered error {e} while converting columns {col} with data type {dtype}")
                spark_dtype = StructField(col, StringType())
            spark_schema.append(spark_dtype)
        return StructType(spark_schema)
