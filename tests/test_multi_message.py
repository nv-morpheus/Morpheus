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

import numpy as np
import pandas as pd
import pytest

import cudf

from morpheus._lib.common import FileTypes
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage
from utils import TEST_DIRS
from utils import create_df_with_dup_ids
from utils import duplicate_df_index
from utils import duplicate_df_index_rand

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

    assert isinstance(request.param, bool), "Indirect parameter needed to be set to use use_cpp"

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

    multi = MultiMessage(meta)

    assert multi.meta is meta
    assert multi.mess_offset == 0
    assert multi.mess_count == meta.count


def test_constructor_values(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta, mess_offset=4, mess_count=5)

    assert multi.meta is meta
    assert multi.mess_offset == 4
    assert multi.mess_count == 5


def test_constructor_invalid(df: cudf.DataFrame):

    meta = MessageMeta(df)

    # Negative offset
    with pytest.raises(ValueError):
        MultiMessage(meta, mess_offset=-1, mess_count=5)

    # Offset beyond start
    with pytest.raises(ValueError):
        MultiMessage(meta, mess_offset=meta.count, mess_count=5)

    # Too large of count
    with pytest.raises(ValueError):
        MultiMessage(meta, mess_offset=0, mess_count=meta.count + 1)

    # Count extends beyond end of dataframe
    with pytest.raises(ValueError):
        MultiMessage(meta, mess_offset=5, mess_count=(meta.count - 5) + 1)


# def test_constructor_invalid_index(config: Config):

#     # Date index
#     with pytest.raises(ValueError):
#         date_index = pd.date_range('1/1/2010', periods=6, freq='D')
#         df = pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]}, index=date_index)

#         MultiMessage(MessageMeta(df))

#     # Categorical index
#     with pytest.raises(ValueError):
#         index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
#         df = pd.DataFrame({
#             'http_status': [200, 200, 404, 404, 301], 'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]
#         },
#                           index=index)

#         MultiMessage(MessageMeta(df))


def test_get_meta(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    # Manually slice the dataframe according to the multi settings
    df_sliced: cudf.DataFrame = df.iloc[multi.mess_offset:multi.mess_offset + multi.mess_count, :]

    assert multi.get_meta().equals(df_sliced)

    # Make sure we return a table here, not a series
    col_name = df_sliced.columns[0]
    assert multi.get_meta(col_name).equals(df_sliced[col_name])

    col_name = [df_sliced.columns[0], df_sliced.columns[2]]
    assert multi.get_meta(col_name).equals(df_sliced[col_name])

    # Out of order columns
    col_name = [df_sliced.columns[3], df_sliced.columns[0]]
    assert multi.get_meta(col_name).equals(df_sliced[col_name])

    # Should fail with missing column
    with pytest.raises(KeyError):
        multi.get_meta("column_that_does_not_exist")


def test_get_meta_dup_index(df: cudf.DataFrame):

    # Duplicate some indices before creating the meta
    df = duplicate_df_index(df, replace_ids={3: 1, 5: 4})

    # Now just run the other test to reuse code
    test_get_meta(df)


def test_set_meta(df: cudf.DataFrame, df_type: typing.Literal['cudf', 'pandas']):

    df_saved = df.copy()

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    saved_mask = np.ones(len(df_saved), bool)
    saved_mask[multi.mess_offset:multi.mess_offset + multi.mess_count] = False

    def compare_df(df_to_check, val_to_check):
        bool_df = df_to_check == val_to_check

        if (df_type == "cudf"):
            bool_df = bool_df.to_pandas()

        return bool(bool_df.all(axis=None))

    def test_value(columns, value):
        multi.set_meta(columns, value)
        assert compare_df(multi.get_meta(columns), value)

        # Now make sure the original dataframe is untouched
        assert compare_df(df_saved[saved_mask], meta.df[saved_mask])

    def test_all_columns(vals_to_set: typing.Iterable):

        vals_to_set = itertools.cycle(vals_to_set)

        curr_val = next(vals_to_set)
        multi.set_meta("v2", curr_val)
        assert compare_df(multi.get_meta("v2"), curr_val)

        curr_val = next(vals_to_set)
        multi.set_meta(["v1", "v3"], curr_val)
        assert compare_df(multi.get_meta(["v1", "v3"]), curr_val)

        curr_val = next(vals_to_set)
        multi.set_meta(["v4", "v2", "v3"], curr_val)
        assert compare_df(multi.get_meta(["v4", "v2", "v3"]), curr_val)

        curr_val = next(vals_to_set)
        multi.set_meta(None, curr_val)
        assert compare_df(multi.get_meta(), curr_val)

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


def test_other():
    # Set from a list
    values = list(range(7))
    multi.set_meta("v2", values)

    meta = MessageMeta(df)
    assert meta.has_unique_index()
    mm = MultiMessage(meta, 0, meta.count)

    mm2 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm2.get_meta()) == 7

    values = list(range(7))
    mm2.set_meta('v2', values)

    assert mm2.get_meta_list('v2') == values

    assert mm2.get_meta_list(None) == mm2.get_meta().to_arrow().to_pylist()


def test_set_meta_new_column(df: cudf.DataFrame):

    meta = MessageMeta(df)

    multi = MultiMessage(meta=meta, mess_offset=3, mess_count=5)

    multi.set_meta("new_column", 10)

    assert (multi.get_meta("new_column") == 10).all()


# @pytest.mark.parametrize('df_type', ['cudf', 'pandas'])
def test_copy_ranges(df_type: typing.Literal['cudf', 'pandas']):
    # if CppConfig.get_should_use_cpp() and df_type == 'pandas':
    #     pytest.skip("Pandas dataframes not supported in C++ mode")

    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type=df_type)

    meta = MessageMeta(df)
    assert meta.has_unique_index()
    assert meta.count == len(df)

    mm = MultiMessage(meta, 0, meta.count)
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == meta.count

    mm2 = mm.copy_ranges([(2, 6)])
    assert len(mm2.meta.df) == 4
    assert mm2.meta.count == 4
    assert len(mm2.get_meta()) == 4
    assert mm2.meta is not meta
    assert mm2.meta.df is not df

    # slice two different ranges of rows
    mm3 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm3.meta.df) == 7
    assert mm3.meta.count == 7
    assert len(mm3.get_meta()) == 7
    assert mm3.meta is not meta
    assert mm3.meta is not mm2.meta
    assert mm3.meta.df is not df
    assert mm3.meta.df is not mm2.meta.df


@pytest.mark.parametrize("dup_row", [0, 1, 8, 18, 19])  # test for dups at the front, middle and the tail
def test_duplicate_ids(tmp_path, dup_row: typing.Literal[0, 1, 8, 18, 19]):
    """
    Test for dataframe with duplicate IDs issue #686
    """
    dup_file = create_df_with_dup_ids(tmp_path, dup_row=dup_row)

    dup_df = read_file_to_df(dup_file, file_type=FileTypes.Auto, df_type='cudf')
    assert not dup_df.index.is_unique

    meta = MessageMeta(dup_df)
    assert not meta.has_unique_index()

    assert meta.count == len(dup_df)

    with meta.mutable_dataframe() as mut_df:
        assert len(mut_df) == len(dup_df)

    # C++ fails here the copy returns 22 rows
    assert len(meta.copy_dataframe()) == len(dup_df)

    mm = MultiMessage(meta, 0, meta.count)

    # Python fails here mm.get_meta_list(None) returns 22 rows
    assert mm.get_meta_list(None) == dup_df.to_arrow().to_pylist()

    meta = MessageMeta(dup_df)
    mm = MultiMessage(meta, 0, len(meta.df))

    # Fails on an assert because len(meta.df) returned 22
    assert mm.get_meta_list(None) == dup_df.to_arrow().to_pylist()
