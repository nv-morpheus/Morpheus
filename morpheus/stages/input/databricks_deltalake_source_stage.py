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

import mrc
import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config

from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from morpheus.stages.utils.databricks_utils import configure_databricks_connect

logger = logging.getLogger(__name__)


@register_stage("from-databricks-deltalake")
class DataBricksDeltaLakeSourceStage(SingleOutputSource):
    """
    Source stage used to load messages from a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    spark_query : str, default None
        SQL Query that need to be executed to fetch the results from deltalake table.
    items_per_page: int, default 1000
        Number of rows per iteration/page to be fetched from remote spark cluster.
    databricks_host : str, default None
        URL of Databricks host to connect to.
    databricks_token : str, default None
        Access token for Databricks cluster.
    databricks_cluster_id : str, default None
        Databricks cluster to be used to query the data as per SQL provided.
    databricks_port : str, default "15001"
        Databricks port that Databricks Connect connects to. Defaults to 15001
    databricks_org_id : str, default "0"
        Azure-only, see ?o=orgId in URL. Defaults to 0 for other platforms
    """

    def __init__(self,
                 config: Config,
                 spark_query: str = None,
                 items_per_page: int = 1000,
                 databricks_host: str = None,
                 databricks_token: str = None,
                 databricks_cluster_id: str = None,
                 databricks_port: str = "15001",
                 databricks_org_id: str = "0"):

        super().__init__(config)
        self.spark_query = spark_query
        configure_databricks_connect(databricks_host,
                                           databricks_token,
                                           databricks_cluster_id,
                                           databricks_port,
                                           databricks_org_id)
        self.spark = SparkSession.builder.getOrCreate()
        self.items_per_page = items_per_page
        self.offset = 0

    @property
    def name(self) -> str:
        return "from-databricks-deltalake"

    def supports_cpp_node(self) -> bool:
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self.source_generator)
        return node, MessageMeta

    def source_generator(self):
        try:
            spark_df = self.spark.sql(self.spark_query)
            spark_df = spark_df.withColumn('_id', sf.monotonically_increasing_id())
            count = spark_df.count()
            while self.offset <= count:
                df = spark_df.where(sf.col('_id').between(self.offset, self.offset + self.items_per_page))
                logger.debug(f"Reading next iteration data between index: {str(self.offset)} and {str(self.offset + self.items_per_page + 1)}")
                self.offset = self.offset + self.items_per_page + 1
                yield MessageMeta(df=cudf.from_pandas(df.toPandas().drop(["_id"],axis=1)))
        except Exception as e:
            logger.exception("Error occurred reading data from feature store and converting to Dataframe: {}".format(e))
            raise Exception(e)
