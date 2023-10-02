import os
from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from unittest import mock
import pytest
from unittest.mock import patch
import cudf
import pandas as pd
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.input.databricks_deltalake_source_stage import DataBricksDeltaLakeSourceStage
import logging

logger = logging.getLogger(__name__)


def test_databricks_deltalake_source_stage_pipe(config: Config, dataset: DatasetManager):
    """
    Test the DataBricksDeltaLakeSourceStage against a mock spark session which will return spark_df converted into a DataFrame with specific rows per page.
    """

    # df = pd.DataFrame([("audit", "system1"),("audit", "system2"),("secure", "system1"),("secure", "system2")], columns=["log","source"])
    expected_df = dataset['filter_probs.csv']
    config = Config()
    with patch('morpheus.stages.input.databricks_deltalake_source_stage.DatabricksSession') as mock_db_session:
        deltaLakeStage = DeltaLakeSourceStage(config,
                     spark_query="",items_per_page=10000,databricks_host="",databricks_token="", databricks_cluster_id="")
        deltaLakeStage.spark.sql.return_value.withColumn.return_value.select.return_value.withColumn.return_value.where.return_value.toPandas.return_value.drop.return_value = expected_df
        deltaLakeStage.spark.sql.return_value.withColumn.return_value.select.return_value.withColumn.return_value.count.return_value = expected_df.shape[0]
        pipe = LinearPipeline(config)
        pipe.set_source(deltaLakeStage)
        comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
        pipe.run()
        assert_results(comp_stage.get_results())
