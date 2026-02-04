# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools

import cudf

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf cimport Column as plc_Column
from pylibcudf cimport Table as plc_Table
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "morpheus/objects/table_info.hpp" namespace "morpheus" nogil:


    cdef cppclass TableInfoData:
        TableInfoData()
        TableInfoData(table_view view,
                      vector[string] indices,
                      vector[string] columns)

        table_view table_view
        vector[string] index_names
        vector[string] column_names
        vector[size_type] column_indices


cdef public api:
    object make_table_from_table_with_metadata(table_with_metadata table, int index_col_count):
        cdef TableWithMetadata tbl_meta = TableWithMetadata.from_libcudf(table)
        df = cudf.DataFrame.from_pylibcudf(tbl_meta)
        if index_col_count > 0:
            df = df.set_index(df.columns[:index_col_count])
        return df

    object make_table_from_table_info_data(TableInfoData table_info, object owner):
        owner_plc_table, _ = owner.reset_index().to_pylibcudf()
        cdef plc_Table view_owner = <plc_Table>owner_plc_table
        cdef plc_Table plc_table = plc_Table.from_table_view(table_info.table_view, view_owner)
        df = cudf.DataFrame.from_pylibcudf(plc_table)
        index_col_count = table_info.index_names.size()
        if index_col_count > 0:
            df = df.set_index(df.columns[:index_col_count])
        return df

    TableInfoData make_table_info_data_from_table(object table):
        cdef plc_Table plc_table = plc_Table(
            [
                col.to_pylibcudf(mode="read")
                for col in itertools.chain(table.index._columns, table._columns)
            ]
        )
        cdef table_view input_table_view = plc_table.view()
        cdef vector[string] index_names
        cdef vector[string] column_names

        # cuDF does a weird check where if there is only one name in both index and columns, and that column is empty or
        # None, then change it to '""'. Not sure what this is used for
        all_names = []
        if isinstance(table.index, cudf.MultiIndex):
            all_names.extend(table.index.names)
        elif table.index.name is not None:
            all_names.append(table.index.name)
        all_names.extend(table.columns)
        check_empty_name = len(all_names) == 1

        for name in table.index.names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            index_names.push_back(str.encode(name))

        for name in table.columns:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            column_names.push_back(str.encode(name))

        return TableInfoData(input_table_view, index_names, column_names)
