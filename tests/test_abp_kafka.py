#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import typing
from io import StringIO
from unittest import mock

import numpy as np
import pandas
import pytest

from morpheus._lib.common import FileTypes
from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.utils import filter_null_data
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.utils.compare_df import compare_df
from utils import TEST_DIRS
from utils import write_file_to_kafka

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer

# End-to-end test intended to imitate the ABP validation test
FEATURE_LENGTH = 29
MODEL_MAX_BATCH_SIZE = 1024


@pytest.mark.kafka
@pytest.mark.slow
@pytest.mark.use_python
@mock.patch('tritonclient.grpc.InferenceServerClient')
def test_abp_no_cpp(mock_triton_client,
                    config: Config,
                    kafka_bootstrap_servers: str,
                    kafka_topics: typing.Tuple[str, str],
                    kafka_consumer: "KafkaConsumer"):
    mock_metadata = {
        "inputs": [{
            'name': 'input__0', 'datatype': 'FP32', "shape": [-1, FEATURE_LENGTH]
        }],
        "outputs": [{
            'name': 'output__0', 'datatype': 'FP32', 'shape': ['-1', '1']
        }]
    }
    mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    data = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'triton_abp_inf_results.csv'), delimiter=',')
    inf_results = np.split(data, range(MODEL_MAX_BATCH_SIZE, len(data), MODEL_MAX_BATCH_SIZE))

    mock_infer_result = mock.MagicMock()
    mock_infer_result.as_numpy.side_effect = inf_results

    def async_infer(callback=None, **k):
        callback(mock_infer_result, None)

    mock_triton_client.async_infer.side_effect = async_infer

    config.mode = PipelineModes.FIL
    config.class_labels = ["mining"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt')) as fh:
        config.fil.feature_columns = [x.strip() for x in fh.readlines()]

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')

    # Fill our topic with the input data
    num_records = write_file_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, val_file_name)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         stop_after=num_records,
                         client_id="test_abp_no_cpp_reader"))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(PreprocessFILStage(config))
    pipe.add_stage(
        TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='test:0000', force_convert_inputs=True))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(
        WriteToKafkaStage(config,
                          bootstrap_servers=kafka_bootstrap_servers,
                          output_topic=kafka_topics.output_topic,
                          client_id="test_abp_no_cpp_writer"))

    pipe.run()

    val_df = read_file_to_df(val_file_name, file_type=FileTypes.Auto, df_type='pandas')

    output_buf = StringIO()
    for rec in kafka_consumer:
        output_buf.write("{}\n".format(rec.value.decode("utf-8")))

    output_buf.seek(0)
    output_df = pandas.read_json(output_buf, lines=True)
    output_df = filter_null_data(output_df)

    assert len(output_df) == num_records

    results = compare_df(val_df, output_df, exclude_columns=[r'^ID$', r'^_ts_'], rel_tol=0.05)

    assert results['diff_rows'] == 0


@pytest.mark.kafka
@pytest.mark.slow
@pytest.mark.use_cpp
@pytest.mark.usefixtures("launch_mock_triton")
def test_abp_cpp(config,
                 kafka_bootstrap_servers: str,
                 kafka_topics: typing.Tuple[str, str],
                 kafka_consumer: "KafkaConsumer"):
    config.mode = PipelineModes.FIL
    config.class_labels = ["mining"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt')) as fh:
        config.fil.feature_columns = [x.strip() for x in fh.readlines()]

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')

    num_records = write_file_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, val_file_name)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         stop_after=num_records))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(PreprocessFILStage(config))

    # We are feeding TritonInferenceStage the port to the grpc server because that is what the validation tests do
    # but the code under-the-hood replaces this with the port number of the http server
    pipe.add_stage(
        TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='localhost:8001',
                             force_convert_inputs=True))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config))
    pipe.add_stage(SerializeStage(config))
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

    assert len(output_df) == num_records

    results = compare_df(val_df, output_df, exclude_columns=[r'^ID$', r'^_ts_'], rel_tol=0.05)

    assert results['diff_rows'] == 0
