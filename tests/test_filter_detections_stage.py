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

from morpheus.common import FilterSource
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage


def _make_message(df, probs):
    df_ = df[0:len(probs)]
    mem = ResponseMemory(count=len(df_), tensors={'probs': probs})
    return MultiResponseMessage(meta=MessageMeta(df_), memory=mem)


def test_constructor(config):
    fds = FilterDetectionsStage(config)
    assert fds.name == "filter"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = fds.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    fds = FilterDetectionsStage(config, threshold=0.2)
    assert fds._controller._threshold == 0.2


@pytest.mark.use_cudf
@pytest.mark.use_python
def test_filter_copy(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_message = _make_message(filter_probs_df, probs)

    # All values are at or below the threshold so nothing should be returned
    output_message = fds._controller.filter_copy(mock_message)
    assert output_message is None

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_message = fds._controller.filter_copy(mock_message)
    assert output_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[1:1, :].to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_message = fds._controller.filter_copy(mock_message)
    assert output_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[2:3, :].to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_message = fds._controller.filter_copy(mock_message)
    mask = cp.zeros(len(filter_probs_df), dtype=cp.bool_)
    mask[2] = True
    mask[4] = True
    assert output_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[mask, :].to_cupy().tolist()


@pytest.mark.use_cudf
@pytest.mark.use_python
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, filter_probs_df, do_copy, threshold, field_name):
    fds = FilterDetectionsStage(config,
                                threshold=threshold,
                                copy=do_copy,
                                filter_source=FilterSource.DATAFRAME,
                                field_name=field_name)
    expected_df = filter_probs_df.to_pandas()
    expected_df = expected_df[expected_df[field_name] > threshold]

    probs = cp.zeros([len(filter_probs_df), 3], 'float')
    mock_message = _make_message(filter_probs_df, probs)

    # All values are at or below the threshold
    output_message = fds._controller.filter_copy(mock_message)

    assert output_message.get_meta().to_cupy().tolist() == expected_df.to_numpy().tolist()


@pytest.mark.use_cudf
@pytest.mark.use_python
def test_filter_slice(config, filter_probs_df):
    fds = FilterDetectionsStage(config, threshold=0.5, filter_source=FilterSource.TENSOR)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_message = _make_message(filter_probs_df, probs)

    # All values are at or below the threshold
    output_messages = fds._controller.filter_slice(mock_message)
    assert len(output_messages) == 0

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_messages = fds._controller.filter_slice(mock_message)
    assert len(output_messages) == 1
    output_message = output_messages[0]
    assert output_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[1:1, :].to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_messages = fds._controller.filter_slice(mock_message)
    assert len(output_messages) == 1
    output_message = output_messages[0]
    assert output_message.offset == 2
    assert output_message.count == 2
    assert output_message.get_meta().to_cupy().tolist() == filter_probs_df.loc[2:3, :].to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(filter_probs_df, probs)

    output_messages = fds._controller.filter_slice(mock_message)
    assert len(output_messages) == 2
    (msg1, msg2) = output_messages  # pylint: disable=unbalanced-tuple-unpacking
    assert msg1.offset == 2
    assert msg1.count == 1

    assert msg2.offset == 4
    assert msg2.count == 1

    assert msg1.get_meta().to_cupy().tolist() == filter_probs_df.loc[2:2, :].to_cupy().tolist()
    assert msg2.get_meta().to_cupy().tolist() == filter_probs_df.loc[4:4, :].to_cupy().tolist()
