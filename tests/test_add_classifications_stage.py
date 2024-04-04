#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.messages import ControlMessage
from morpheus.config import Config
from morpheus._lib.messages import TensorMemory as CppTensorMemory
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage


@pytest.fixture(name="config")
def config_fixture(config: Config):
    config.class_labels = ['frogs', 'lizards', 'toads']
    yield config


def test_constructor(config: Config):
    stage = AddClassificationsStage(config)
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['frogs', 'lizards', 'toads']
    assert stage._idx2label == {0: 'frogs', 1: 'lizards', 2: 'toads'}
    assert stage.name == "add-class"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


def test_constructor_explicit_labels(config: Config):
    stage = AddClassificationsStage(config, threshold=1.3, labels=['lizards'], prefix='test_')
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['lizards']
    assert stage._idx2label == {1: 'test_lizards'}


def test_constructor_errors(config: Config):
    with pytest.raises(AssertionError):
        AddClassificationsStage(config, labels=['missing'])


@pytest.mark.use_python
def test_add_labels_with_multi_response_message_and_contgrol_message():

    class_labels = {0: "frogs", 1: "lizards", 2: "toads"}

    threshold = 0.6

    df = cudf.DataFrame([0, 1], columns=["dummy"])
    probs_array = cp.array([[0.1, 0.6, 0.8], [0.3, 0.61, 0.9]])
    probs_array_bool = probs_array > threshold

    mrm = MultiResponseMessage(meta=MessageMeta(df), memory=TensorMemory(count=2, tensors={"probs": probs_array}))

    labeled_mrm = AddClassificationsStage._add_labels(mrm, idx2label=class_labels, threshold=threshold)

    DatasetManager.assert_df_equal(labeled_mrm.get_meta("frogs"), probs_array_bool[:, 0])
    DatasetManager.assert_df_equal(labeled_mrm.get_meta("lizards"), probs_array_bool[:, 1])
    DatasetManager.assert_df_equal(labeled_mrm.get_meta("toads"), probs_array_bool[:, 2])

    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(CppTensorMemory(count=2, tensors={"probs": probs_array}))
    
    labeled_cm = AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=threshold)
    
    # Check that the labeled control message and labeled multi response message are the same
    DatasetManager.assert_df_equal(labeled_cm.get_meta("frogs"), labeled_mrm.get_meta("frogs"))
    DatasetManager.assert_df_equal(labeled_cm.get_meta("lizards"), labeled_mrm.get_meta("lizards"))
    DatasetManager.assert_df_equal(labeled_cm.get_meta("toads"), labeled_mrm.get_meta("toads"))

    # Same thing but change the probs tensor name
    mrm = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"other_probs": probs_array}),
                                   probs_tensor_name="other_probs")

    labeled_mrm = AddClassificationsStage._add_labels(mrm, idx2label=class_labels, threshold=threshold)

    DatasetManager.assert_df_equal(labeled_mrm.get_meta("frogs"), probs_array_bool[:, 0])
    DatasetManager.assert_df_equal(labeled_mrm.get_meta("lizards"), probs_array_bool[:, 1])
    DatasetManager.assert_df_equal(labeled_mrm.get_meta("toads"), probs_array_bool[:, 2])

    # Fail in missing probs data
    mrm = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"other_probs": probs_array}),
                                   probs_tensor_name="other_probs")
    mrm.probs_tensor_name = "probs"

    with pytest.raises(KeyError):
        AddClassificationsStage._add_labels(mrm, idx2label=class_labels, threshold=threshold)

    # Too small of a probs array
    mrm = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"probs": probs_array[:, 0:-1]}))

    with pytest.raises(RuntimeError):
        AddClassificationsStage._add_labels(mrm, idx2label=class_labels, threshold=threshold)

    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(CppTensorMemory(count=2, tensors={"probs": probs_array[:, 0:-1]}))

    with pytest.raises(RuntimeError):
        AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=threshold)
    