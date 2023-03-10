#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
import os
import queue
from unittest import mock

import numpy as np
import pytest

from morpheus.config import ConfigFIL
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.inference.triton_inference_stage import ResourcePool
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage

MODEL_MAX_BATCH_SIZE = 1024


def test_resource_pool():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(2)

    rp = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert rp._queue.qsize() == 0

    # Check for normal allocation
    assert rp.borrow_obj() == 0
    assert rp._queue.qsize() == 0
    assert rp.added_count == 1
    create_fn.assert_called_once()

    assert rp.borrow_obj() == 1
    assert rp._queue.qsize() == 0
    assert rp.added_count == 2
    assert create_fn.call_count == 2

    rp.return_obj(0)
    assert rp._queue.qsize() == 1
    rp.return_obj(1)
    assert rp._queue.qsize() == 2

    assert rp.borrow_obj() == 0
    assert rp._queue.qsize() == 1
    assert rp._added_count == 2
    assert create_fn.call_count == 2

    assert rp.borrow_obj() == 1
    assert rp._queue.qsize() == 0
    assert rp._added_count == 2
    assert create_fn.call_count == 2


def test_resource_pool_overallocate():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(5)

    rp = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert rp.borrow_obj() == 0
    assert rp.borrow_obj() == 1

    with pytest.raises(queue.Empty):
        rp.borrow_obj(timeout=0)


def test_resource_pool_large_count():
    create_fn = mock.MagicMock()
    create_fn.side_effect = range(10000)

    rp = ResourcePool[int](create_fn=create_fn, max_size=10000)

    for _ in range(10000):
        rp.borrow_obj(timeout=0)

    assert rp._queue.qsize() == 0
    assert create_fn.call_count == 10000


def test_resource_pool_create_raises_error():
    create_fn = mock.MagicMock()
    create_fn.side_effect = (10, RuntimeError, 20)

    rp = ResourcePool[int](create_fn=create_fn, max_size=10)

    assert rp.borrow_obj() == 10

    with pytest.raises(RuntimeError):
        rp.borrow_obj()

    assert rp.borrow_obj() == 20


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.parametrize('num_records', [1000, 2000, 4000])
@mock.patch('tritonclient.grpc.InferenceServerClient')
def test_triton_stage_pipe(mock_triton_client, config, tmp_path, num_records):
    mock_metadata = {
        "inputs": [{
            'name': 'input__0', 'datatype': 'FP32', "shape": [-1, 1]
        }],
        "outputs": [{
            'name': 'output__0', 'datatype': 'FP32', 'shape': ['-1', '1']
        }]
    }
    mock_model_config = {"config": {"max_batch_size": MODEL_MAX_BATCH_SIZE}}

    input_file = os.path.join(tmp_path, "input_data.csv")
    with open(input_file, 'w') as fh:
        writer = csv.writer(fh, dialect=csv.excel)
        writer.writerow(['v'])
        for i in range(num_records):
            writer.writerow([i * 2])

    mock_triton_client.return_value = mock_triton_client
    mock_triton_client.is_server_live.return_value = True
    mock_triton_client.is_server_ready.return_value = True
    mock_triton_client.is_model_ready.return_value = True
    mock_triton_client.get_model_metadata.return_value = mock_metadata
    mock_triton_client.get_model_config.return_value = mock_model_config

    data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    inf_results = np.split(data, range(MODEL_MAX_BATCH_SIZE, len(data), MODEL_MAX_BATCH_SIZE))

    mock_infer_result = mock.MagicMock()
    mock_infer_result.as_numpy.side_effect = inf_results

    def async_infer(callback=None, **k):
        callback(mock_infer_result, None)

    mock_triton_client.async_infer.side_effect = async_infer

    config.mode = PipelineModes.FIL
    config.class_labels = ["test"]
    config.model_max_batch_size = MODEL_MAX_BATCH_SIZE
    config.pipeline_batch_size = 1024
    config.feature_length = 1
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.fil = ConfigFIL()

    config.fil.feature_columns = ['v']

    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(PreprocessFILStage(config))
    pipe.add_stage(
        TritonInferenceStage(config, model_name='abp-nvsmi-xgb', server_url='test:0000', force_convert_inputs=True))
    pipe.add_stage(AddScoresStage(config, prefix="score_"))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    results = np.loadtxt(out_file, delimiter=',', skiprows=1)
    assert len(results) == num_records

    for (i, row) in enumerate(results):
        assert (row == [i, i * 2, i * 2]).all()
