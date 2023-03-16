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

# pylint: disable=redefined-outer-name

import os
import typing

import cupy as cp
import numpy as np
import pytest

import cudf

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.memory.response_memory import ResponseMemory
from morpheus.messages.memory.response_memory import ResponseMemoryProbs
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.messages.multi_inference_ae_message import MultiInferenceAEMessage
from morpheus.messages.multi_inference_message import MultiInferenceFILMessage
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_inference_message import MultiInferenceNLPMessage
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.messages.multi_response_message import MultiResponseProbsMessage
from morpheus.messages.multi_tensor_message import MultiTensorMessage
from utils import TEST_DIRS
from utils import assert_df_equal
from utils import duplicate_df_index
from utils import duplicate_df_index_rand


@pytest.fixture(scope="function")
def df(df_type: typing.Literal['cudf', 'pandas'], use_cpp: bool):

    return read_file_to_df(os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv'),
                           file_type=FileTypes.Auto,
                           df_type=df_type)


def test_constructor_empty(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta)

    assert multi.meta is meta
    assert multi.mess_offset == 0
    assert multi.mess_count == meta.count


def test_constructor_values(df: cudf.DataFrame):

    meta = MessageMeta(df)

    # No count
    multi = MultiMessage(meta=meta, mess_offset=2)
    assert multi.meta is meta
    assert multi.mess_offset == 2
    assert multi.mess_count == meta.count - multi.mess_offset

    # No offset
    multi = MultiMessage(meta=meta, mess_count=9)
    assert multi.meta is meta
    assert multi.mess_offset == 0
    assert multi.mess_count == 9

    # Both
    multi = MultiMessage(meta=meta, mess_offset=4, mess_count=5)
    assert multi.meta is meta
    assert multi.mess_offset == 4
    assert multi.mess_count == 5


def test_constructor_invalid(df: cudf.DataFrame):

    meta = MessageMeta(df)

    # Negative offset
    with pytest.raises(ValueError):
        MultiMessage(meta=meta, mess_offset=-1, mess_count=5)

    # Offset beyond start
    with pytest.raises(ValueError):
        MultiMessage(meta=meta, mess_offset=meta.count, mess_count=5)

    # Too large of count
    with pytest.raises(ValueError):
        MultiMessage(meta=meta, mess_offset=0, mess_count=meta.count + 1)

    # Count extends beyond end of dataframe
    with pytest.raises(ValueError):
        MultiMessage(meta=meta, mess_offset=5, mess_count=(meta.count - 5) + 1)


def test_get_meta(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    # Manually slice the dataframe according to the multi settings
    df_sliced: cudf.DataFrame = df.iloc[multi.mess_offset:multi.mess_offset + multi.mess_count, :]

    assert assert_df_equal(multi.get_meta(), df_sliced)

    # Make sure we return a table here, not a series
    col_name = df_sliced.columns[0]
    assert assert_df_equal(multi.get_meta(col_name), df_sliced[col_name])

    col_name = [df_sliced.columns[0], df_sliced.columns[2]]
    assert assert_df_equal(multi.get_meta(col_name), df_sliced[col_name])

    # Out of order columns
    col_name = [df_sliced.columns[3], df_sliced.columns[0]]
    assert assert_df_equal(multi.get_meta(col_name), df_sliced[col_name])

    # Should fail with missing column
    with pytest.raises(KeyError):
        multi.get_meta("column_that_does_not_exist")

    # Finally, check that we dont overwrite the original dataframe
    multi.get_meta(col_name).iloc[:] = 5
    # assert not assert_df_equal(df_sliced[col_name], 5.0)
    assert assert_df_equal(multi.get_meta(col_name), df_sliced[col_name])


def test_get_meta_dup_index(df: cudf.DataFrame):

    # Duplicate some indices before creating the meta
    df = duplicate_df_index(df, replace_ids={3: 1, 5: 4})

    # Now just run the other test to reuse code
    test_get_meta(df)


def test_set_meta(df: cudf.DataFrame, df_type: typing.Literal['cudf', 'pandas']):

    df_saved = df.copy()

    if (df_type == "cudf"):
        df_saved = df_saved.to_pandas()

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    saved_mask = np.ones(len(df_saved), bool)
    saved_mask[multi.mess_offset:multi.mess_offset + multi.mess_count] = False

    def test_value(columns, value):
        multi.set_meta(columns, value)
        assert assert_df_equal(multi.get_meta(columns), value)

        # Now make sure the original dataframe is untouched
        assert assert_df_equal(df_saved[saved_mask], meta.df[saved_mask])

    single_column = "v2"
    two_columns = ["v1", "v3"]
    multi_columns = ["v4", "v2", "v3"]  # out of order as well

    # Setting an integer
    test_value(None, 0)
    test_value(single_column, 1)
    test_value(two_columns, 2)
    test_value(multi_columns, 3)

    # Setting a list (Single column only)
    test_value(single_column, list(range(0, 0 + multi.mess_count)))

    # Setting numpy arrays (single column)
    test_value(None, np.random.randn(multi.mess_count, 1))
    test_value(single_column, np.random.randn(multi.mess_count))  # Must be single dimension
    test_value(two_columns, np.random.randn(multi.mess_count, 1))
    test_value(multi_columns, np.random.randn(multi.mess_count, 1))

    # Setting numpy arrays (multi column)
    test_value(None, np.random.randn(multi.mess_count, len(df.columns)))
    test_value(two_columns, np.random.randn(multi.mess_count, len(two_columns)))
    test_value(multi_columns, np.random.randn(multi.mess_count, len(multi_columns)))


def test_set_meta_new_column(df: cudf.DataFrame, df_type: typing.Literal['cudf', 'pandas']):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    # Just one new column
    val_to_set = list(range(multi.mess_count))
    multi.set_meta("new_column", val_to_set)
    assert assert_df_equal(multi.get_meta("new_column"), val_to_set)

    if (df_type == "cudf"):
        # cudf isnt capable of setting more than one new column at a time
        return

    # Now set one with new and old columns
    val_to_set = np.random.randn(multi.mess_count, 2)
    multi.set_meta(["v2", "new_column2"], val_to_set)
    assert assert_df_equal(multi.get_meta(["v2", "new_column2"]), val_to_set)


def test_set_meta_new_column_dup_index(df: cudf.DataFrame, df_type: typing.Literal['cudf', 'pandas']):

    # Duplicate some indices before creating the meta
    df = duplicate_df_index(df, replace_ids={3: 4, 5: 4})

    test_set_meta_new_column(df, df_type)


def test_copy_ranges(df: cudf.DataFrame):

    meta = MessageMeta(df)

    mm = MultiMessage(meta=meta)

    mm2 = mm.copy_ranges([(2, 6)])
    assert len(mm2.meta.df) == 4
    assert mm2.meta.count == 4
    assert len(mm2.get_meta()) == 4
    assert mm2.meta is not meta
    assert mm2.meta.df is not df
    assert mm2.mess_offset == 0
    assert mm2.mess_count == 6 - 2
    assert assert_df_equal(mm2.get_meta(), df.iloc[2:6])

    # slice two different ranges of rows
    mm3 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm3.meta.df) == 7
    assert mm3.meta.count == 7
    assert len(mm3.get_meta()) == 7
    assert mm3.meta is not meta
    assert mm3.meta is not mm2.meta
    assert mm3.meta.df is not df
    assert mm3.meta.df is not mm2.meta.df
    assert mm3.mess_offset == 0
    assert mm3.mess_count == (6 - 2) + (15 - 12)
    assert assert_df_equal(mm3.get_meta(), df.iloc[2:6].append(df.iloc[12:15]))


def test_copy_ranges_dup_index(df: cudf.DataFrame):

    # Duplicate some indices before creating the meta
    df = duplicate_df_index_rand(df, count=4)

    # Now just run the other test to reuse code
    test_copy_ranges(df)


def test_get_slice_ranges(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi_full = MultiMessage(meta=meta)

    # Get the whole thing
    slice1 = multi_full.get_slice(multi_full.mess_offset, multi_full.mess_count)
    assert slice1.meta is meta
    assert slice1.mess_offset == slice1.mess_offset
    assert slice1.mess_count == slice1.mess_count

    # Smaller slice
    slice2 = multi_full.get_slice(2, 18)
    assert slice2.mess_offset == 2
    assert slice2.mess_count == 18 - 2

    # Chained slice
    slice4 = multi_full.get_slice(3, 19).get_slice(1, 10)
    assert slice4.mess_offset == 3 + 1
    assert slice4.mess_count == 10 - 1

    # Negative start
    with pytest.raises(IndexError):
        multi_full.get_slice(-1, multi_full.mess_count)

    # Past the end
    with pytest.raises(IndexError):
        multi_full.get_slice(0, multi_full.mess_count + 1)

    # Stop before start
    with pytest.raises(IndexError):
        multi_full.get_slice(5, 4)

    # Empty slice
    with pytest.raises(IndexError):
        multi_full.get_slice(5, 5)

    # Offset + Count past end
    with pytest.raises(IndexError):
        multi_full.get_slice(13, 13 + (multi_full.mess_count - 13) + 1)

    # Invalid chain, stop past end
    with pytest.raises(IndexError):
        multi_full.get_slice(13, 16).get_slice(1, 5)

    # Invalid chain, start past end
    with pytest.raises(IndexError):
        multi_full.get_slice(13, 16).get_slice(4, 5)


def test_get_slice_values(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi_full = MultiMessage(meta=meta)

    # Single slice
    assert assert_df_equal(multi_full.get_slice(3, 8).get_meta(), df.iloc[3:8])

    # Single slice with one columns
    assert assert_df_equal(multi_full.get_slice(3, 8).get_meta("v1"), df.iloc[3:8]["v1"])

    # Single slice with multiple columns
    assert assert_df_equal(multi_full.get_slice(3, 8).get_meta(["v4", "v3", "v1"]), df.iloc[3:8][["v4", "v3", "v1"]])

    # Chained slice
    assert assert_df_equal(multi_full.get_slice(2, 18).get_slice(5, 9).get_meta(), df.iloc[2 + 5:(2 + 5) + (9 - 5)])

    # Chained slice one column
    assert assert_df_equal(
        multi_full.get_slice(2, 18).get_slice(5, 9).get_meta("v1"), df.iloc[2 + 5:(2 + 5) + (9 - 5)]["v1"])

    # Chained slice multi column
    assert assert_df_equal(
        multi_full.get_slice(2, 18).get_slice(5, 9).get_meta(["v4", "v3", "v1"]),
        df.iloc[2 + 5:(2 + 5) + (9 - 5)][["v4", "v3", "v1"]])

    # Set values
    multi_full.get_slice(4, 10).set_meta(None, 1.15)
    assert assert_df_equal(multi_full.get_slice(4, 10).get_meta(), df.iloc[4:10])

    # Set values one column
    multi_full.get_slice(1, 6).set_meta("v3", 5.3)
    assert assert_df_equal(multi_full.get_slice(1, 6).get_meta("v3"), df.iloc[1:6]["v3"])

    # Set values multi column
    multi_full.get_slice(5, 8).set_meta(["v4", "v1", "v3"], 7)
    assert assert_df_equal(multi_full.get_slice(5, 8).get_meta(["v4", "v1", "v3"]), df.iloc[5:8][["v4", "v1", "v3"]])

    # Chained Set values
    multi_full.get_slice(10, 20).get_slice(1, 4).set_meta(None, 8)
    assert assert_df_equal(multi_full.get_slice(10, 20).get_slice(1, 4).get_meta(), df.iloc[10 + 1:(10 + 1) + (4 - 1)])

    # Chained Set values one column
    multi_full.get_slice(10, 20).get_slice(3, 5).set_meta("v4", 112)
    assert assert_df_equal(
        multi_full.get_slice(10, 20).get_slice(3, 5).get_meta("v4"), df.iloc[10 + 3:(10 + 3) + (5 - 3)]["v4"])

    # Chained Set values multi column
    multi_full.get_slice(10, 20).get_slice(5, 8).set_meta(["v4", "v1", "v2"], 22)
    assert assert_df_equal(
        multi_full.get_slice(10, 20).get_slice(5, 8).get_meta(["v4", "v1", "v2"]),
        df.iloc[10 + 5:(10 + 5) + (8 - 5)][["v4", "v1", "v2"]])


def test_get_slice_values_dup_index(df: cudf.DataFrame):

    # Duplicate some indices before creating the meta
    df = duplicate_df_index_rand(df, count=4)

    # Now just run the other test to reuse code
    test_get_slice_values(df)


def test_get_slice_derived(df: cudf.DataFrame):

    multi_tensor_message_tensors = {
        "input_ids": cp.zeros((20, 2)),
        "input_mask": cp.zeros((20, 2)),
        "seq_ids": cp.expand_dims(cp.arange(0, 20, dtype=int), axis=1),
        "input__0": cp.zeros((20, 2)),
        "probs": cp.zeros((20, 2)),
    }

    def compare_slice(message_class, **kwargs):
        multi = message_class(**kwargs)
        assert isinstance(multi.get_slice(0, 20), message_class)

    meta = MessageMeta(df)

    # Base MultiMessages
    compare_slice(MultiMessage, meta=meta)
    compare_slice(MultiAEMessage, meta=meta, model=None, train_scores_mean=0.0, train_scores_std=1.0)

    # Tensor messages
    compare_slice(MultiTensorMessage, meta=meta, memory=TensorMemory(count=20, tensors=multi_tensor_message_tensors))

    # Inference messages
    compare_slice(MultiInferenceMessage,
                  meta=meta,
                  memory=InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceNLPMessage,
                  meta=meta,
                  memory=InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceFILMessage,
                  meta=meta,
                  memory=InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceAEMessage,
                  meta=meta,
                  memory=InferenceMemory(count=20, tensors=multi_tensor_message_tensors))

    # Response messages
    compare_slice(MultiResponseMessage,
                  meta=meta,
                  memory=ResponseMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiResponseProbsMessage,
                  meta=meta,
                  memory=ResponseMemoryProbs(count=20, probs=multi_tensor_message_tensors["probs"]))


def test_from_message(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=10)

    multi_tensor_message_tensors = {
        "input_ids": cp.zeros((20, 2)),
        "input_mask": cp.zeros((20, 2)),
        "seq_ids": cp.expand_dims(cp.arange(0, 20, dtype=int), axis=1),
        "input__0": cp.zeros((20, 2)),
        "probs": cp.zeros((20, 2)),
    }

    # Once for the base multi-message class
    multi2 = MultiMessage.from_message(multi)
    assert multi2.meta is multi.meta
    assert multi2.mess_offset == multi.mess_offset
    assert multi2.mess_count == multi.mess_count

    multi2 = MultiMessage.from_message(multi, mess_offset=5)
    assert multi2.meta is multi.meta
    assert multi2.mess_offset == 5
    assert multi2.mess_count == multi.mess_count

    multi2 = MultiMessage.from_message(multi, mess_count=7)
    assert multi2.meta is multi.meta
    assert multi2.mess_offset == multi.mess_offset
    assert multi2.mess_count == 7

    multi2 = MultiMessage.from_message(multi, mess_offset=6, mess_count=9)
    assert multi2.meta is multi.meta
    assert multi2.mess_offset == 6
    assert multi2.mess_count == 9

    meta2 = MessageMeta(df[7:14])
    multi2 = MultiMessage.from_message(multi, meta=meta2)
    assert multi2.meta is meta2
    assert multi2.mess_offset == 0
    assert multi2.mess_count == meta2.count

    multi2 = MultiMessage.from_message(multi, meta=meta2, mess_offset=4)
    assert multi2.meta is meta2
    assert multi2.mess_offset == 4
    assert multi2.mess_count == meta2.count - 4

    multi2 = MultiMessage.from_message(multi, meta=meta2, mess_count=4)
    assert multi2.meta is meta2
    assert multi2.mess_offset == 0
    assert multi2.mess_count == 4

    # Repeat for tensor memory
    memory = ResponseMemory(count=20, tensors=multi_tensor_message_tensors)
    multi_tensor = MultiTensorMessage(meta=meta, mess_offset=3, mess_count=10, memory=memory, offset=5, count=10)

    # Create from a base class
    multi3: MultiTensorMessage = MultiTensorMessage.from_message(multi, memory=memory)
    assert multi3.memory is memory
    assert multi3.offset == 0
    assert multi3.count == memory.count

    # Create from existing instance
    multi3 = MultiTensorMessage.from_message(multi_tensor)
    assert multi3.memory is memory
    assert multi3.offset == multi_tensor.offset
    assert multi3.count == multi_tensor.count

    multi3 = MultiTensorMessage.from_message(multi_tensor, offset=5)
    assert multi3.memory is memory
    assert multi3.offset == 5
    assert multi3.count == multi_tensor.count

    multi3 = MultiTensorMessage.from_message(multi_tensor, count=12)
    assert multi3.memory is memory
    assert multi3.offset == multi_tensor.offset
    assert multi3.count == 12

    multi3 = MultiTensorMessage.from_message(multi_tensor, offset=7, count=9)
    assert multi3.memory is memory
    assert multi3.offset == 7
    assert multi3.count == 9

    memory3 = ResponseMemory(count=20, tensors=multi_tensor_message_tensors)
    multi3 = MultiTensorMessage.from_message(multi_tensor, memory=memory3)
    assert multi3.memory is memory3
    assert multi3.offset == 0
    assert multi3.count == memory3.count

    multi3 = MultiTensorMessage.from_message(multi_tensor, memory=memory3, offset=2)
    assert multi3.memory is memory3
    assert multi3.offset == 2
    assert multi3.count == memory3.count - 2

    multi3 = MultiTensorMessage.from_message(multi_tensor, memory=memory3, count=14)
    assert multi3.memory is memory3
    assert multi3.offset == 0
    assert multi3.count == 14

    multi3 = MultiTensorMessage.from_message(multi_tensor, memory=memory3, offset=9, count=8)
    assert multi3.memory is memory3
    assert multi3.offset == 9
    assert multi3.count == 8

    # Test missing memory
    with pytest.raises(AttributeError):
        MultiTensorMessage.from_message(multi)

    # Finally, test a class with extra arguments
    multi4 = MultiAEMessage.from_message(multi, model=None, train_scores_mean=0.0, train_scores_std=1.0)
    assert multi4.meta is meta
    assert multi4.mess_offset == multi.mess_offset
    assert multi4.mess_count == multi.mess_count

    multi5 = MultiAEMessage.from_message(multi4)
    assert multi5.model is multi4.model
    assert multi5.train_scores_mean == multi4.train_scores_mean
    assert multi5.train_scores_std == multi4.train_scores_std

    multi5 = MultiAEMessage.from_message(multi4, train_scores_mean=7.0)
    assert multi5.model is multi4.model
    assert multi5.train_scores_mean == 7.0
    assert multi5.train_scores_std == multi4.train_scores_std

    # Test missing other options
    with pytest.raises(AttributeError):
        MultiAEMessage.from_message(multi)


def test_tensor_constructor(df: cudf.DataFrame):

    mess_len = len(df)

    multi_tensor_message_tensors = {
        "input_ids": cp.zeros((mess_len, 2)),
        "input_mask": cp.zeros((mess_len, 2)),
        "seq_ids": cp.expand_dims(cp.arange(0, mess_len, dtype=int), axis=1),
        "input__0": cp.zeros((mess_len, 2)),
        "probs": cp.zeros((mess_len, 2)),
    }

    meta = MessageMeta(df)

    memory = ResponseMemory(count=mess_len, tensors=multi_tensor_message_tensors)

    # Default constructor
    multi_tensor = MultiTensorMessage(meta=meta, memory=memory)
    assert multi_tensor.meta is meta
    assert multi_tensor.mess_offset == 0
    assert multi_tensor.mess_count == meta.count
    assert multi_tensor.memory is memory
    assert multi_tensor.offset == 0
    assert multi_tensor.count == memory.count

    # All constructor values
    multi_tensor = MultiTensorMessage(meta=meta, mess_offset=3, mess_count=5, memory=memory, offset=5, count=10)
    assert multi_tensor.meta is meta
    assert multi_tensor.mess_offset == 3
    assert multi_tensor.mess_count == 5
    assert multi_tensor.memory is memory
    assert multi_tensor.offset == 5
    assert multi_tensor.count == 10

    # Larger tensor count
    multi_tensor = MultiTensorMessage(meta=meta,
                                      memory=TensorMemory(count=17, tensors={"probs": cp.random.rand(17, 2)}))
    assert multi_tensor.meta is meta
    assert multi_tensor.mess_offset == 0
    assert multi_tensor.mess_count == meta.count
    assert multi_tensor.memory is memory
    assert multi_tensor.offset == 0
    assert multi_tensor.count == memory.count

    # Negative offset
    with pytest.raises(ValueError):
        MultiTensorMessage(meta=meta, memory=memory, offset=-1)

    # Offset beyond start
    with pytest.raises(ValueError):
        MultiTensorMessage(meta=meta, memory=memory, offset=memory.count, count=5)

    # Too large of count
    with pytest.raises(ValueError):
        MultiTensorMessage(meta=meta, memory=memory, offset=0, count=memory.count + 1)

    # Count extends beyond end of dataframe
    with pytest.raises(ValueError):
        MultiTensorMessage(meta=meta, memory=memory, offset=5, count=(memory.count - 5) + 1)

    # Count smaller than mess_count
    with pytest.raises(ValueError):
        MultiTensorMessage(meta=meta, mess_count=10, memory=memory, count=9)


def test_tensor_slicing(df: cudf.DataFrame):

    mess_len = len(df)

    repeat_counts = [1] * mess_len
    repeat_counts[1] = 2
    repeat_counts[4] = 5
    repeat_counts[5] = 3
    repeat_counts[7] = 6
    tensor_count = sum(repeat_counts)

    probs = cp.random.rand(tensor_count, 2)
    seq_ids = cp.zeros((tensor_count, 3), dtype=cp.int32)

    for i, r in enumerate(repeat_counts):
        seq_ids[sum(repeat_counts[:i]):sum(repeat_counts[:i]) + r] = cp.ones((r, 3), int) * i

    # First with no offsets
    memory = InferenceMemory(count=tensor_count, tensors={"seq_ids": seq_ids, "probs": probs})
    multi = MultiInferenceMessage(meta=MessageMeta(df), memory=memory)
    multi_slice = multi.get_slice(3, 10)
    assert multi_slice.mess_offset == seq_ids[3, 0].item()
    assert multi_slice.mess_count == seq_ids[10, 0].item() - seq_ids[3, 0].item()
    assert multi_slice.offset == 3
    assert multi_slice.count == 10 - 3
    assert cp.all(multi_slice.get_tensor("probs") == probs[3:10, :])

    # Offset on memory
    multi = MultiInferenceMessage(meta=MessageMeta(df), memory=memory, offset=4)
    multi_slice = multi.get_slice(6, 13)
    assert multi_slice.mess_offset == seq_ids[multi.offset + 6, 0].item()
    assert multi_slice.mess_count == seq_ids[multi.offset + 13 - 1, 0].item() + 1 - seq_ids[multi.offset + 6, 0].item()
    assert multi_slice.offset == 6 + 4
    assert multi_slice.count == 13 - 6
    assert cp.all(multi_slice.get_tensor("probs") == probs[multi.offset + 6:multi.offset + 13, :])

    # Should be equivalent to shifting the input tensors and having no offset
    equiv_memory = InferenceMemory(count=tensor_count - 4, tensors={"seq_ids": seq_ids[4:], "probs": probs[4:]})
    equiv_multi = MultiInferenceMessage(meta=MessageMeta(df), memory=equiv_memory)
    equiv_slice = equiv_multi.get_slice(6, 13)
    assert multi_slice.mess_offset == equiv_slice.mess_offset
    assert multi_slice.mess_count == equiv_slice.mess_count
    assert multi_slice.offset != equiv_slice.offset
    assert multi_slice.count == equiv_slice.count
    assert cp.all(multi_slice.get_tensor("probs") == equiv_slice.get_tensor("probs"))

    # Offset on meta
    multi = MultiInferenceMessage(meta=MessageMeta(df), mess_offset=3, memory=memory)
    multi_slice = multi.get_slice(2, 9)
    assert multi_slice.mess_offset == seq_ids[multi.offset + 2, 0].item() + 3
    assert multi_slice.mess_count == seq_ids[multi.offset + 9 - 1, 0].item() + 1 - seq_ids[multi.offset + 2, 0].item()
    assert multi_slice.offset == 2
    assert multi_slice.count == 9 - 2
    assert cp.all(multi_slice.get_tensor("probs") == probs[multi.offset + 2:multi.offset + 9, :])

    # Should be equivalent to shifting the input dataframe and having no offset
    equiv_memory = InferenceMemory(count=tensor_count, tensors={"seq_ids": seq_ids, "probs": probs})
    equiv_multi = MultiInferenceMessage(meta=MessageMeta(df.iloc[3:, :]), memory=equiv_memory)
    equiv_slice = equiv_multi.get_slice(2, 9)
    assert multi_slice.mess_offset == equiv_slice.mess_offset + 3
    assert multi_slice.mess_count == equiv_slice.mess_count
    assert multi_slice.offset == equiv_slice.offset
    assert multi_slice.count == equiv_slice.count
    assert assert_df_equal(multi_slice.get_meta(), equiv_slice.get_meta())
