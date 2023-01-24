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
import os
from mrc.core import operators as ops

import mrc
import cudf
import typing
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.messages import MessageMeta
from morpheus.io import serializers


from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


logger = logging.getLogger(__name__)


@register_stage("to-deltalake")
class DeltaLakeSinkStage(SinglePortStage):
    """
    Sink stage used to write messages to a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    delta_path : str
        Path of the delta table where the data need to be written or updated.
    databricks_host : str
        URL of Databricks host to connect to.
    databricks_token : str
        Access token for Databricks cluster.
    databricks_cluster_id : str
        Databricks cluster to be used to query the data as per SQL provided.
    databricks_port : str
        Databricks port that Databricks Connect connects to. Defaults to 15001
    databricks_org_id : str
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
        self._configure_databricks_connect(databricks_host, databricks_token, databricks_cluster_id, databricks_port,databricks_org_id)
        self.spark = SparkSession.builder.config("spark.databricks.delta.optimizeWrite.enabled", "true")\
        .config("spark.databricks.delta.autoCompact.enabled", "true")\
        .getOrCreate()

        # Enable Arrow-based columnar data transfers
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


    @property
    def name(self) -> str:
        return "to-delta"

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
            def write_to_deltalake(x: MessageMeta):
                # convert cudf to spark dataframe
                df = x.df.to_pandas()
                schema = self._extract_schema_from_pandas_dataframe(df)
                spark_df = self.spark.createDataFrame(df,schema=schema)
                spark_df.write.format('delta')\
                .option("mergeSchema", "true").mode("append")\
                .save(self.delta_path)
                return x
            obs.pipe(ops.map(write_to_deltalake)).subscribe(sub)

        to_delta = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(stream, to_delta)
        stream = to_delta

        # Return input unchanged to allow passthrough
        return stream, input_stream[1]

    def _extract_schema_from_pandas_dataframe(self, df):
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


    def _configure_databricks_connect(self,databricks_host, databricks_token, databricks_cluster_id, databricks_port, databricks_org_id):
        if(os.environ.get('DATABRICKS_HOST',None)==None and databricks_host==None):
            raise Exception("Parameter for databricks host not provided")
        if(os.environ.get('DATABRICKS_TOKEN',None)==None and databricks_token==None):
            raise Exception("Parameter for databricks token not provided")
        if(os.environ.get('DATABRICKS_CLUSTER_ID',None)==None and databricks_cluster_id==None):
            raise Exception("Parameter for databricks cluster not provided")
        host = None
        cluster = None
        token = None
        config_file = "/root/.databricks-connect"
        should_add = False
        if(os.environ.get('DATABRICKS_HOST',None)!=None):
            host = os.environ.get('DATABRICKS_HOST')
        else:
            host = databricks_host
        if(os.environ.get('DATABRICKS_TOKEN',None)!=None):
            token = os.environ.get('DATABRICKS_TOKEN')
        else:
            token = databricks_token
        if(os.environ.get('DATABRICKS_CLUSTER_ID',None)!=None):
            cluster = os.environ.get('DATABRICKS_CLUSTER_ID')
        else:
            cluster = databricks_cluster_id
        config = """{
                      "host": "@host",
                      "token": "@token",
                      "cluster_id": "@cluster_id",
                      "org_id": "@org_id",
                      "port": "@port"
                }"""
        config = config.replace("@host",host).replace("@token",token).replace("@cluster_id",cluster).replace("@org_id",databricks_org_id).replace("@port",databricks_port)

        # check if the config file for databricks connect already exists
        config_exist = os.path.exists(config_file)
        if config_exist:
            # check if the config being added already exists, if so do nothing
            with open(config_file) as f:
                if config in f.read():
                    logger.info("Configuration for databricks-connect already exists, nothing added!")
                else:
                    logger.info("Configuration not found for databricks-connect, adding provided configs!")
                    should_add = True
        else:
            should_add = True
        if should_add:
            with open(config_file,"w+") as f:
                f.write(config)
            logger.info("Databricks-connect successfully configured!")
