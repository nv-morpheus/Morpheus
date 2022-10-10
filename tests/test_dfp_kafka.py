#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import typing
from io import StringIO

import pandas
import pytest

from morpheus._lib.file_types import FileTypes
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.utils import filter_null_data
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.auto_encoder_inference_stage import AutoEncoderInferenceStage
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess import train_ae_stage
from morpheus.utils.compare_df import compare_df
from morpheus.utils.logger import configure_logging
from utils import TEST_DIRS

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer

configure_logging(log_level=logging.DEBUG)
# End-to-end test intended to imitate the dfp validation test


@pytest.mark.kafka
@pytest.mark.slow
@pytest.mark.use_python
def test_dfp_roleg(config,
                   kafka_bootstrap_servers: str,
                   kafka_topics: typing.Tuple[str, str],
                   kafka_consumer: "KafkaConsumer"):

    config.mode = PipelineModes.AE
    config.mode = PipelineModes.AE
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.num_threads = 1
    config.use_cpp = False

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "role-g"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-*-input.csv")
    train_data_glob = os.path.join(TEST_DIRS.tests_data_dir, "dfp-cloudtrail-*-training-data.csv")
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-role-g-validation-data-output.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(
        train_ae_stage.TrainAEStage(
            config,
            train_data_glob=train_data_glob,
            source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
            seed=42,
            sort_glob=True))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(
        WriteToKafkaStage(config, bootstrap_servers=kafka_bootstrap_servers, output_topic=kafka_topics.output_topic))

    pipe.run()

    val_df = read_file_to_df(val_file_name, file_type=FileTypes.Auto, df_type='pandas')

    output_buf = StringIO()
    for rec in kafka_consumer:
        output_buf.write("{}\n".format(rec.value.decode("utf-8")))

    output_buf.seek(0)
    output_df = pandas.read_json(output_buf, lines=True)
    output_df = filter_null_data(output_df)

    assert len(output_df) == len(val_df)

    results = compare_df(val_df,
                         output_df,
                         replace_idx="_index_",
                         include_columns=["mean_abs_z", "max_abs_z", ".*_z_loss", ".*_loss", ".*_pred"],
                         rel_tol=0.15,
                         show_report=True)

    assert results['diff_rows'] == 0


@pytest.mark.kafka
@pytest.mark.slow
@pytest.mark.use_python
def test_dfp_user123(config,
                     kafka_bootstrap_servers: str,
                     kafka_topics: typing.Tuple[str, str],
                     kafka_consumer: "KafkaConsumer"):

    config.mode = PipelineModes.AE
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "user123"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-*-input.csv")
    train_data_glob = os.path.join(TEST_DIRS.tests_data_dir, "dfp-cloudtrail-*-training-data.csv")
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-user123-validation-data-output.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(
        train_ae_stage.TrainAEStage(
            config,
            train_data_glob=train_data_glob,
            source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
            seed=42,
            sort_glob=True))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(
        WriteToKafkaStage(config, bootstrap_servers=kafka_bootstrap_servers, output_topic=kafka_topics.output_topic))

    pipe.run()

    val_df = read_file_to_df(val_file_name, file_type=FileTypes.Auto, df_type='pandas')

    output_buf = StringIO()
    for rec in kafka_consumer:
        output_buf.write("{}\n".format(rec.value.decode("utf-8")))

    output_buf.seek(0)
    output_df = pandas.read_json(output_buf, lines=True)
    output_df = filter_null_data(output_df)

    assert len(output_df) == len(val_df)

    results = compare_df(val_df,
                         output_df,
                         replace_idx="_index_",
                         include_columns=["mean_abs_z", "max_abs_z", ".*_loss", ".*_z_loss"],
                         rel_tol=0.1,
                         show_report=True)

    assert results['diff_rows'] == 0
