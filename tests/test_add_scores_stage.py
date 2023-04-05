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

import cupy as cp
import pytest

import cudf

from dataset_loader import DatasetLoader
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage


def test_constructor(config):
    config.class_labels = ['frogs', 'lizards', 'toads']
    config.feature_length = 12

    a = AddScoresStage(config)
    assert a._class_labels == ['frogs', 'lizards', 'toads']
    assert a._labels == ['frogs', 'lizards', 'toads']
    assert a._idx2label == {0: 'frogs', 1: 'lizards', 2: 'toads'}
    assert a.name == "add-scores"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = a.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    a = AddScoresStage(config, labels=['lizards'], prefix='test_')
    assert a._class_labels == ['frogs', 'lizards', 'toads']
    assert a._labels == ['lizards']
    assert a._idx2label == {1: 'test_lizards'}

    with pytest.raises(AssertionError):
        AddScoresStage(config, labels=['missing'])


@pytest.mark.use_python
def test_add_labels():
    class_labels = {0: "frogs", 1: "lizards", 2: "toads"}

    df = cudf.DataFrame([0, 1], columns=["dummy"])
    probs_array = cp.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])

    message = MultiResponseMessage(meta=MessageMeta(df), memory=TensorMemory(count=2, tensors={"probs": probs_array}))

    labeled = AddClassificationsStage._add_labels(message, idx2label=class_labels, threshold=None)

    assert DatasetLoader.assert_df_equal(labeled.get_meta("frogs"), probs_array[:, 0])
    assert DatasetLoader.assert_df_equal(labeled.get_meta("lizards"), probs_array[:, 1])
    assert DatasetLoader.assert_df_equal(labeled.get_meta("toads"), probs_array[:, 2])

    # Same thing but change the probs tensor name
    message = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"other_probs": probs_array}),
                                   probs_tensor_name="other_probs")

    labeled = AddClassificationsStage._add_labels(message, idx2label=class_labels, threshold=None)

    assert DatasetLoader.assert_df_equal(labeled.get_meta("frogs"), probs_array[:, 0])
    assert DatasetLoader.assert_df_equal(labeled.get_meta("lizards"), probs_array[:, 1])
    assert DatasetLoader.assert_df_equal(labeled.get_meta("toads"), probs_array[:, 2])

    # Fail in missing probs data
    message = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"other_probs": probs_array}),
                                   probs_tensor_name="other_probs")
    message.probs_tensor_name = "probs"

    with pytest.raises(KeyError):
        AddClassificationsStage._add_labels(message, idx2label=class_labels, threshold=None)

    # Too small of a probs array
    message = MultiResponseMessage(meta=MessageMeta(df),
                                   memory=TensorMemory(count=2, tensors={"probs": probs_array[:, 0:-1]}))

    with pytest.raises(RuntimeError):
        AddClassificationsStage._add_labels(message, idx2label=class_labels, threshold=None)
