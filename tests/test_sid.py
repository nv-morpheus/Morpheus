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
from unittest import mock

import numpy as np
import pytest
import srf

from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages.multi_response_message import MultiResponseProbsMessage
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from utils import TEST_DIRS
from utils import calc_error_val
from utils import compare_class_to_scores

# End-to-end test intended to imitate the Sid validation test
FEATURE_LENGTH = 256
MODEL_MAX_BATCH_SIZE = 32


class LambdaMapStage(SinglePortStage):
    """
    This class writes messages to an s3 bucket.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bucket: strWE
        Name of the s3 bucket to write to.

    """

    def __init__(self, c: Config, s3_writer):
        super().__init__(c)

        self._s3_writer = s3_writer

    @property
    def name(self) -> str:
        return "to-s3-bucket"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.messages.message_meta.MessageMeta`, )
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        node = builder.make_node(self.unique_name, self._s3_writer)
        builder.make_edge(stream, node)

        stream = node

        # Return input unchanged to allow passthrough
        return stream, input_stream[1]


@pytest.mark.slow
@pytest.mark.use_python
@mock.patch('tritonclient.grpc.InferenceServerClient')
def test_minibert_no_cpp(mock_triton_client, config, tmp_path):
    mock_metadata = {
        "inputs": [{
            "name": "input_ids", "datatype": "INT32", "shape": [-1, FEATURE_LENGTH]
        }, {
            "name": "attention_mask", "datatype": "INT32", "shape": [-1, FEATURE_LENGTH]
        }],
        "outputs": [{
            "name": "output", "datatype": "FP32", "shape": [-1, 10]
        }]
    }
    mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    data = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'triton_sid_inf_results.csv'), delimiter=',')
    inf_results = np.split(data, range(MODEL_MAX_BATCH_SIZE, len(data), MODEL_MAX_BATCH_SIZE))

    mock_infer_result = mock.MagicMock()
    mock_infer_result.as_numpy.side_effect = inf_results

    def async_infer(callback=None, **k):
        callback(mock_infer_result, None)

    mock_triton_client.async_infer.side_effect = async_infer

    config.mode = PipelineModes.NLP
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'sid-validation-data.csv')
    vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file_name,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))
    pipe.add_stage(
        TritonInferenceStage(config, model_name='sid-minibert-onnx', server_url='fake:001', force_convert_inputs=True))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config, threshold=0.5, prefix="si_"))
    pipe.add_stage(AddScoresStage(config, prefix="score_"))
    pipe.add_stage(
        ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()
    results = calc_error_val(results_file_name)

    compare_class_to_scores(out_file, config.class_labels, 'si_', 'score_', threshold=0.5)
    assert results.diff_rows == 1333


def _run_minibert_cpp(config, tmp_path, model_name, truncated):
    config.mode = PipelineModes.NLP
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = FEATURE_LENGTH
    config.edge_buffer_size = 128
    config.num_threads = 1

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'sid-validation-data.csv')
    vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file_name,
                           truncation=truncated,
                           do_lower_case=True,
                           add_special_tokens=False))
    pipe.add_stage(
        TritonInferenceStage(config, model_name=model_name, server_url='localhost:8001', force_convert_inputs=True))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config, threshold=0.5, prefix="si_"))
    pipe.add_stage(AddScoresStage(config, prefix="score_"))
    pipe.add_stage(
        ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()
    compare_class_to_scores(out_file, config.class_labels, 'si_', 'score_', threshold=0.5)
    return calc_error_val(results_file_name)


@pytest.mark.slow
@pytest.mark.use_cpp
@pytest.mark.usefixtures("launch_mock_triton")
def test_minibert_cpp_truncated(config, tmp_path):
    results = _run_minibert_cpp(config, tmp_path, 'sid-minibert-onnx', True)
    assert results.diff_rows == 1204


@pytest.mark.slow
@pytest.mark.use_cpp
@pytest.mark.usefixtures("launch_mock_triton")
def test_minibert_cpp(config, tmp_path):
    results = _run_minibert_cpp(config, tmp_path, 'sid-minibert-onnx-no-trunc', False)
    assert results.diff_rows == 18


@pytest.mark.slow
@pytest.mark.use_cpp
def test_drift(config):

    root_dir = "/home/mdemoret/Repos/morpheus/data/deloitte/lm-morph"

    num_threads = 6
    pipeline_batch_size = 524288
    model_max_batch_size = 524288
    model_seq_length = 6
    column_headers_file = os.path.join(root_dir, "data/column_headers.txt")
    input_file = os.path.join(root_dir, "data/test/test_lm.csv")
    model_name = "drift-onnx"
    server_url = "localhost:8001"
    output_file = os.path.join(root_dir, "output_filter_cpp.csv")

    # CppConfig.set_should_use_cpp(False)

    config = Config()
    config.log_level = logging.DEBUG
    config.fil = ConfigFIL()
    config.mode = PipelineModes.FIL
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_seq_length
    config.class_labels = ["score"]

    with open(column_headers_file, "r") as lf:
        config.fil.feature_columns = [x.strip() for x in lf.readlines()]
        print("Loaded columns. Current columns: ", str(config.fil.feature_columns))

    # Create a pipeline object
    pipeline = LinearPipeline(config)

    # Add a source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False, repeat=1))
    pipeline.add_stage(MonitorStage(config, description="File source rate"))

    # Add a deserialize and pre-process stage
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PreprocessFILStage(config))
    pipeline.add_stage(MonitorStage(config, description="Pre-process rate"))

    pipeline.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=server_url,
                             force_convert_inputs=True,
                             use_shared_memory=False))
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit="inf"))

    # pipeline.add_stage(TriggerStage(config))

    pipeline.add_stage(AddScoresStage(config))

    pipeline.add_stage(FilterDetectionsStage(config, threshold=0.5, copy=True))
    # # pipeline.add_stage(MonitorStage(config, description="Filter rate"))

    # def post_scores(x: MultiResponseProbsMessage):
    #     # Do stuff
    #     scores = x.get_meta("score")

    #     print("got scores")

    #     return x

    # pipeline.add_stage(LambdaMapStage(config, post_scores))

    # pipeline.add_stage(TriggerStage(config))

    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Run the pipeline
    pipeline.run()
