#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import cudf

from _utils import assert_results
from _utils import mk_async_infer
from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.inference.triton_inference_stage import ProducerConsumerQueue
from morpheus.stages.inference.triton_inference_stage import ResourcePool
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceWorker
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage

MODEL_MAX_BATCH_SIZE = 1024


def test_resource_pool():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(2)

    pool = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert pool._queue.qsize() == 0

    # Check for normal allocation
    assert pool.borrow_obj() == 0
    assert pool._queue.qsize() == 0
    assert pool.added_count == 1
    create_fn.assert_called_once()

    assert pool.borrow_obj() == 1
    assert pool._queue.qsize() == 0
    assert pool.added_count == 2
    assert create_fn.call_count == 2

    pool.return_obj(0)
    assert pool._queue.qsize() == 1
    pool.return_obj(1)
    assert pool._queue.qsize() == 2

    assert pool.borrow_obj() == 0
    assert pool._queue.qsize() == 1
    assert pool._added_count == 2
    assert create_fn.call_count == 2

    assert pool.borrow_obj() == 1
    assert pool._queue.qsize() == 0
    assert pool._added_count == 2
    assert create_fn.call_count == 2


def test_resource_pool_overallocate():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(5)

    pool = ResourcePool[int](create_fn=create_fn, max_size=2)

    assert pool.borrow_obj() == 0
    assert pool.borrow_obj() == 1

    with pytest.raises(queue.Empty):
        pool.borrow_obj(timeout=0)


def test_resource_pool_large_count():
    create_fn = mock.MagicMock()
    create_fn.side_effect = range(10000)

    pool = ResourcePool[int](create_fn=create_fn, max_size=10000)

    for _ in range(10000):
        pool.borrow_obj(timeout=0)

    assert pool._queue.qsize() == 0
    assert create_fn.call_count == 10000


def test_resource_pool_create_raises_error():
    create_fn = mock.MagicMock()
    create_fn.side_effect = (10, RuntimeError, 20)

    pool = ResourcePool[int](create_fn=create_fn, max_size=10)

    assert pool.borrow_obj() == 10

    with pytest.raises(RuntimeError):
        pool.borrow_obj()

    assert pool.borrow_obj() == 20


@pytest.mark.skip(reason="TODO: determine what to do about python impls")
@pytest.mark.cpu_mode
@pytest.mark.parametrize("pipeline_mode", list(PipelineModes))
def test_stage_constructor_worker_class(config: Config, pipeline_mode: PipelineModes):
    config.mode = pipeline_mode
    stage = TritonInferenceStage(config, model_name='test', server_url='test:0000')
    worker = stage._get_inference_worker(ProducerConsumerQueue())
    assert isinstance(worker, TritonInferenceWorker)


@pytest.mark.skip(reason="TODO: determine what to do about python impls")
@pytest.mark.cpu_mode
@pytest.mark.parametrize("pipeline_mode", list(PipelineModes))
@pytest.mark.parametrize("needs_logits", [True, False, None])
def test_stage_get_inference_worker(config: Config, pipeline_mode: PipelineModes, needs_logits: bool | None):
    if needs_logits is None:
        expexted_needs_logits = (pipeline_mode == PipelineModes.NLP)
    else:
        expexted_needs_logits = needs_logits

    config.mode = pipeline_mode

    stage = TritonInferenceStage(config, model_name='test', server_url='test:0000', needs_logits=needs_logits)

    worker = stage._get_inference_worker(ProducerConsumerQueue())
    assert isinstance(worker, TritonInferenceWorker)
    assert worker.needs_logits == expexted_needs_logits
