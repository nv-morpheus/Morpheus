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

import itertools
import os
import typing

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf

from morpheus._lib.common import FileTypes
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.memory import inference_memory
from morpheus.messages.memory import response_memory
from morpheus.messages.memory import tensor_memory
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
from utils import create_df_with_dup_ids
from utils import duplicate_df_index
from utils import duplicate_df_index_rand


def compare_df(df_to_check, val_to_check):
    # Comparisons work better in cudf so convert everything to that
    if (isinstance(df_to_check, cudf.DataFrame) or isinstance(df_to_check, cudf.Series)):
        df_to_check = df_to_check.to_pandas()

    if (isinstance(val_to_check, cudf.DataFrame) or isinstance(val_to_check, cudf.Series)):
        val_to_check = val_to_check.to_pandas()

    bool_df = df_to_check == val_to_check

    return bool(bool_df.all(axis=None))


# def pytest_generate_tests(metafunc: pytest.Metafunc):
#     """
#     This function will add parameterizations for the `config` fixture depending on what types of config the test
#     supports
#     """

#     # Only care about the config fixture
#     if ("df" not in metafunc.fixturenames):
#         return

#     use_pandas = metafunc.definition.get_closest_marker("use_pandas") is not None
#     use_cudf = metafunc.definition.get_closest_marker("use_cudf") is not None

#     if (use_pandas and use_cudf):
#         raise RuntimeError(
#             "Both markers (use_pandas and use_cudf) were added to function {}. Remove markers to support both.".format(
#                 metafunc.definition.nodeid))
#     elif (not use_pandas and not use_cudf):
#         # See if we are marked with cpp or python
#         use_cpp = metafunc.definition.get_closest_marker("use_cpp") is not None
#         use_python = metafunc.definition.get_closest_marker("use_python") is not None

#         if (not use_cpp and not use_python):
#             # Add the 3 parameters directly
#             metafunc.parametrize(
#                 "df",
#                 [
#                     pytest.param(
#                         "pandas", marks=[pytest.mark.use_pandas, pytest.mark.use_python], id="use_pandas-use_python"),
#                     pytest.param("cudf", marks=[pytest.mark.use_cudf, pytest.mark.use_python],
#                                  id="use_cudf-use_python"),
#                     pytest.param("cudf", marks=[pytest.mark.use_cudf, pytest.mark.use_cpp], id="use_cudf-use_cpp")
#                 ],
#                 indirect=True)
#         else:
#             # Add the markers to the parameters
#             metafunc.parametrize("df",
#                                  [
#                                      pytest.param("pandas", marks=pytest.mark.use_pandas, id="use_pandas"),
#                                      pytest.param("cudf", marks=pytest.mark.use_cudf, id="use_cudf")
#                                  ],
#                                  indirect=True)


def pytest_generate_tests(metafunc: pytest.Metafunc):

    # Only care about the df fixture
    if ("df" not in metafunc.fixturenames):
        return

    # Since we need a dataframe, lets create the parameters for it (which are not an inner product)
    metafunc.parametrize("df_type,use_cpp",
                         [
                             pytest.param("pandas", False, id="use_pandas-use_python"),
                             pytest.param("cudf", False, id="use_cudf-use_python"),
                             pytest.param("cudf", True, id="use_cudf-use_cpp")
                         ],
                         indirect=True)


# Autouse this fixture since each test in this file should use C++ and Python
@pytest.fixture(scope="function", autouse=True)
def use_cpp(request: pytest.FixtureRequest):

    assert hasattr(request, "param") and isinstance(request.param, bool), "Indirect parameter needed to be set to use use_cpp"

    do_use_cpp: bool = request.param

    CppConfig.set_should_use_cpp(do_use_cpp)

    yield do_use_cpp


@pytest.fixture(scope="function")
def df_type(request: pytest.FixtureRequest):

    assert request.param in ["pandas", "cudf"], "Indirect parameter needed to be set to use df_type"

    df_type_str: typing.Literal["cudf", "pandas"] = request.param

    yield df_type_str


@pytest.fixture(scope="function")
def df(df_type: typing.Literal['cudf', 'pandas']):

    # if (not hasattr(request, "param")):
    #     use_pandas = request.node.get_closest_marker("use_pandas") is not None
    #     use_cudf = request.node.get_closest_marker("use_cudf") is not None

    #     assert use_pandas != use_cudf, "Invalid config"

    #     df_type = "pandas" if use_pandas else "cudf"

    # else:
    #     assert request.param in ["pandas", "cudf"]

    #     df_type = request.param

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

    assert compare_df(multi.get_meta(), df_sliced)

    # Make sure we return a table here, not a series
    col_name = df_sliced.columns[0]
    assert compare_df(multi.get_meta(col_name), df_sliced[col_name])

    col_name = [df_sliced.columns[0], df_sliced.columns[2]]
    assert compare_df(multi.get_meta(col_name), df_sliced[col_name])

    # Out of order columns
    col_name = [df_sliced.columns[3], df_sliced.columns[0]]
    assert compare_df(multi.get_meta(col_name), df_sliced[col_name])

    # Should fail with missing column
    with pytest.raises(KeyError):
        multi.get_meta("column_that_does_not_exist")

    # Finally, check that we dont overwrite the original dataframe
    multi.get_meta(col_name).iloc[:] = 5
    # assert not compare_df(df_sliced[col_name], 5.0)
    assert compare_df(multi.get_meta(col_name), df_sliced[col_name])


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
        assert compare_df(multi.get_meta(columns), value)

        # Now make sure the original dataframe is untouched
        assert compare_df(df_saved[saved_mask], meta.df[saved_mask])

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
    assert compare_df(multi.get_meta("new_column"), val_to_set)

    if (df_type == "cudf"):
        # cudf isnt capable of setting more than one new column at a time
        return

    # Now set one with new and old columns
    val_to_set = np.random.randn(multi.mess_count, 2)
    multi.set_meta(["v2", "new_column2"], val_to_set)
    assert compare_df(multi.get_meta(["v2", "new_column2"]), val_to_set)


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
    assert compare_df(mm2.get_meta(), df.iloc[2:6])

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
    assert compare_df(mm3.get_meta(), df.iloc[2:6].append(df.iloc[12:15]))


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

    # # Single slice
    # assert compare_df(multi_full.get_slice(3, 8).get_meta(), df.iloc[3:8])

    # # Single slice with one columns
    # assert compare_df(multi_full.get_slice(3, 8).get_meta("v1"), df.iloc[3:8]["v1"])

    # # Single slice with multiple columns
    # assert compare_df(multi_full.get_slice(3, 8).get_meta(["v4", "v3", "v1"]), df.iloc[3:8][["v4", "v3", "v1"]])

    # # Chained slice
    # assert compare_df(multi_full.get_slice(2, 18).get_slice(5, 9).get_meta(), df.iloc[2 + 5:(2 + 5) + (9 - 5)])

    # # Chained slice one column
    # assert compare_df(
    #     multi_full.get_slice(2, 18).get_slice(5, 9).get_meta("v1"), df.iloc[2 + 5:(2 + 5) + (9 - 5)]["v1"])

    # # Chained slice multi column
    # assert compare_df(
    #     multi_full.get_slice(2, 18).get_slice(5, 9).get_meta(["v4", "v3", "v1"]),
    #     df.iloc[2 + 5:(2 + 5) + (9 - 5)][["v4", "v3", "v1"]])

    # Set values
    multi_full.get_slice(4, 10).set_meta(None, 1.15)
    assert compare_df(multi_full.get_slice(4, 10).get_meta(), df.iloc[4:10])

    # Set values one column
    multi_full.get_slice(1, 6).set_meta("v3", 5.3)
    assert compare_df(multi_full.get_slice(1, 6).get_meta("v3"), df.iloc[1:6]["v3"])

    # Set values multi column
    multi_full.get_slice(5, 8).set_meta(["v4", "v1", "v3"], 7)
    assert compare_df(multi_full.get_slice(5, 8).get_meta(["v4", "v1", "v3"]), df.iloc[5:8][["v4", "v1", "v3"]])

    # Chained Set values
    multi_full.get_slice(10, 20).get_slice(1, 4).set_meta(None, 8)
    assert compare_df(multi_full.get_slice(10, 20).get_slice(1, 4).get_meta(), df.iloc[10 + 1:(10 + 1) + (4 - 1)])

    # Chained Set values one column
    multi_full.get_slice(10, 20).get_slice(3, 5).set_meta("v4", 112)
    assert compare_df(
        multi_full.get_slice(10, 20).get_slice(3, 5).get_meta("v4"), df.iloc[10 + 3:(10 + 3) + (5 - 3)]["v4"])

    # Chained Set values multi column
    multi_full.get_slice(10, 20).get_slice(5, 8).set_meta(["v4", "v1", "v2"], 22)
    assert compare_df(
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
    compare_slice(MultiAEMessage, meta=meta)

    # Tensor messages
    compare_slice(MultiTensorMessage,
                  meta=meta,
                  memory=tensor_memory.TensorMemory(count=20, tensors=multi_tensor_message_tensors))

    # Inference messages
    compare_slice(MultiInferenceMessage,
                  meta=meta,
                  memory=inference_memory.InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceNLPMessage,
                  meta=meta,
                  memory=inference_memory.InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceFILMessage,
                  meta=meta,
                  memory=inference_memory.InferenceMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiInferenceAEMessage,
                  meta=meta,
                  memory=inference_memory.InferenceMemory(count=20, tensors=multi_tensor_message_tensors))

    # Response messages
    compare_slice(MultiResponseMessage,
                  meta=meta,
                  memory=response_memory.ResponseMemory(count=20, tensors=multi_tensor_message_tensors))
    compare_slice(MultiResponseProbsMessage,
                  meta=meta,
                  memory=response_memory.ResponseMemoryProbs(count=20, probs=multi_tensor_message_tensors["probs"]))
