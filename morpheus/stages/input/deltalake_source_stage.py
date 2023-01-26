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

import mrc
from pyspark.sql import SparkSession

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("from-deltalake")
class DeltaLakeSourceStage(SingleOutputSource):
    """
    Source stage used to load messages from a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    spark_query : str
        SQL Query that need to be executed to fetch the results from deltalake table.
    databricks_host : str
        URL of Databricks host to connect to.
    databricks_token : str
        Access token for Databricks cluster.
    databricks_cluster_id : str
        Databricks cluster to be used to query the data as per SQL provided.
    databricks_port : str
        Databricks port that Databricks Connect connects to. Defaults to 15001
    databricks_org_id : str
        Azure-only, see ?o=orgId in URL. Defaults to 0 for other platforms
    """

    def __init__(self,
                 config: Config,
                 spark_query: str,
                 databricks_host: str = None,
                 databricks_token: str = None,
                 databricks_cluster_id: str = None,
                 databricks_port: str = "15001",
                 databricks_org_id: str = "0"):

        super().__init__(config)
        self.spark_query = spark_query
        self._configure_databricks_connect(databricks_host,
                                           databricks_token,
                                           databricks_cluster_id,
                                           databricks_port,
                                           databricks_org_id)
        self.spark = SparkSession.builder.getOrCreate()

    @property
    def name(self) -> str:
        return "from-delta"

    def supports_cpp_node(self) -> bool:
        return False

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self.source_generator)
        return node, MessageMeta

    def source_generator(self):
        try:
            yield MessageMeta(df=cudf.from_pandas(self.spark.sql(self.spark_query).toPandas()))
        except Exception as e:
            logger.exception("Error occurred reading data from feature store and converting to Dataframe: {}".format(e))
            raise Exception(e)

    def _configure_databricks_connect(self,
                                      databricks_host,
                                      databricks_token,
                                      databricks_cluster_id,
                                      databricks_port,
                                      databricks_org_id):
        if (os.environ.get('DATABRICKS_HOST', None) is None and databricks_host is None):
            raise Exception("Parameter for databricks host not provided")
        if (os.environ.get('DATABRICKS_TOKEN', None) is None and databricks_token is None):
            raise Exception("Parameter for databricks token not provided")
        if (os.environ.get('DATABRICKS_CLUSTER_ID', None) is None and databricks_cluster_id is None):
            raise Exception("Parameter for databricks cluster not provided")
        host = None
        cluster = None
        token = None
        config_file = "/root/.databricks-connect"
        should_add = False
        config = """{
                      "host": "@host",
                      "token": "@token",
                      "cluster_id": "@cluster_id",
                      "org_id": "@org_id",
                      "port": "@port"
                }"""
        if (os.environ.get('DATABRICKS_HOST', None) is not None):
            host = os.environ.get('DATABRICKS_HOST')
        else:
            host = databricks_host
        if (os.environ.get('DATABRICKS_TOKEN', None) is not None):
            token = os.environ.get('DATABRICKS_TOKEN')
        else:
            token = databricks_token
        if (os.environ.get('DATABRICKS_CLUSTER_ID', None) is not None):
            cluster = os.environ.get('DATABRICKS_CLUSTER_ID')
        else:
            cluster = databricks_cluster_id

        config = config.replace("@host", host).replace("@token", token).replace("@cluster_id", cluster).replace(
            "@org_id", databricks_org_id).replace("@port", databricks_port)

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
            with open(config_file, "w+") as f:
                f.write(config)
            logger.info("Databricks-connect successfully configured!")
