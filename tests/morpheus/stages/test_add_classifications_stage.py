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

import typing

import cupy as cp
import pytest
import typing_utils

import cudf

from _utils.dataset_manager import DatasetManager
# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import TensorMemory as CppTensorMemory
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage


@pytest.fixture(name="config")
def config_fixture(config: Config, use_cpp: bool):  # pylint: disable=unused-argument
    config.class_labels = ['frogs', 'lizards', 'toads']
    yield config


def test_constructor(config: Config):
    stage = AddClassificationsStage(config)
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['frogs', 'lizards', 'toads']
    assert stage._idx2label == {0: 'frogs', 1: 'lizards', 2: 'toads'}
    assert stage.name == "add-class"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


def test_constructor_explicit_labels(config: Config):
    stage = AddClassificationsStage(config, threshold=1.3, labels=['lizards'], prefix='test_')
    assert stage._class_labels == ['frogs', 'lizards', 'toads']
    assert stage._labels == ['lizards']
    assert stage._idx2label == {1: 'test_lizards'}


def test_constructor_errors(config: Config):
    with pytest.raises(AssertionError):
        AddClassificationsStage(config, labels=['missing'])


@pytest.mark.use_python
def test_add_labels():

    class_labels = {0: "frogs", 1: "lizards", 2: "toads"}

    threshold = 0.6

    df = cudf.DataFrame([0, 1], columns=["dummy"])
    probs_array = cp.array([[0.1, 0.6, 0.8], [0.3, 0.61, 0.9]])
    probs_array_bool = probs_array > threshold

    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(CppTensorMemory(count=2, tensors={"probs": probs_array}))

    labeled_cm = AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=threshold)

    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("frogs"), probs_array_bool[:, 0])
    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("lizards"), probs_array_bool[:, 1])
    DatasetManager.assert_df_equal(labeled_cm.payload().get_data("toads"), probs_array_bool[:, 2])

    # Too small of a probs array
    cm = ControlMessage()
    cm.payload(MessageMeta(df))
    cm.tensors(CppTensorMemory(count=2, tensors={"probs": probs_array[:, 0:-1]}))

    with pytest.raises(RuntimeError):
        AddClassificationsStage._add_labels(cm, idx2label=class_labels, threshold=threshold)
