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
from unittest import mock

import cupy as cp
import pytest

from morpheus._lib.file_types import FileTypes
from morpheus._lib.filter_source import FilterSource
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MultiResponseProbsMessage
from morpheus.messages import ResponseMemoryProbs
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from utils import TEST_DIRS


def _make_message(df, probs):
    df_ = df[0:len(probs)]
    mem = ResponseMemoryProbs(count=len(df_), probs=probs)
    return MultiResponseProbsMessage(MessageMeta(df_), 0, len(df_), mem, 0, len(df_))


def test_constructor(config):
    fds = FilterDetectionsStage(config)
    assert fds.name == "filter"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = fds.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    fds = FilterDetectionsStage(config, threshold=0.2)
    assert fds._threshold == 0.2


@pytest.mark.use_python
def test_filter_copy(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')

    fds = FilterDetectionsStage(config, threshold=0.5)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_message = _make_message(df, probs)

    # All values are at or below the threshold
    output_message = fds.filter_copy(mock_message)
    assert len(output_message.get_meta()) == 0

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_message = fds.filter_copy(mock_message)
    assert output_message.get_meta().to_cupy().tolist() == df.loc[1:1, :].to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_message = fds.filter_copy(mock_message)
    assert output_message.get_meta().to_cupy().tolist() == df.loc[2:3, :].to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_message = fds.filter_copy(mock_message)
    mask = cp.zeros(len(df), dtype=cp.bool_)
    mask[2] = True
    mask[4] = True
    assert output_message.get_meta().to_cupy().tolist() == df.loc[mask, :].to_cupy().tolist()


@pytest.mark.use_python
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, do_copy, threshold, field_name):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')

    fds = FilterDetectionsStage(config,
                                threshold=threshold,
                                copy=do_copy,
                                filter_source=FilterSource.DATAFRAME,
                                field_name=field_name)

    probs = cp.zeros([len(df), 3], 'float')
    mock_message = _make_message(df, probs)

    # All values are at or below the threshold
    output_message = fds.filter_copy(mock_message)

    expected_df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='pandas')
    expected_df = expected_df[expected_df[field_name] > threshold]
    output_message.get_meta().to_cupy().tolist() == expected_df.to_numpy().tolist()


@pytest.mark.use_python
def test_filter_slice(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')

    fds = FilterDetectionsStage(config, threshold=0.5)

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_message = _make_message(df, probs)

    # All values are at or below the threshold
    output_messages = fds.filter_slice(mock_message)
    assert len(output_messages) == 0

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_messages = fds.filter_slice(mock_message)
    assert len(output_messages) == 1
    output_message = output_messages[0]
    assert output_message.get_meta().to_cupy().tolist() == df.loc[1:1, :].to_cupy().tolist()

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_messages = fds.filter_slice(mock_message)
    assert len(output_messages) == 1
    output_message = output_messages[0]
    assert output_message.offset == 2
    assert output_message.count == 2
    assert output_message.get_meta().to_cupy().tolist() == df.loc[2:3, :].to_cupy().tolist()

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = _make_message(df, probs)

    output_messages = fds.filter_slice(mock_message)
    assert len(output_messages) == 2
    (m1, m2) = output_messages
    assert m1.offset == 2
    assert m1.count == 1

    assert m2.offset == 4
    assert m2.count == 1

    assert m1.get_meta().to_cupy().tolist() == df.loc[2:2, :].to_cupy().tolist()
    assert m2.get_meta().to_cupy().tolist() == df.loc[4:4, :].to_cupy().tolist()


@pytest.mark.use_python
def test_build_single(config):
    mock_stream = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_builder.make_node.return_value = mock_stream
    mock_input = mock.MagicMock()

    fds = FilterDetectionsStage(config)
    fds._build_single(mock_builder, mock_input)

    mock_builder.make_node.assert_called_once()
    mock_builder.make_edge.assert_called_once()
