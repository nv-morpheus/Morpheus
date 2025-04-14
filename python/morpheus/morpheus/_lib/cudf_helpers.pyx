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
from cudf.core.column import ColumnBase

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf cimport Column as plc_Column
from pylibcudf cimport Table as plc_Table
from pylibcudf.libcudf.io.types cimport column_name_info
from pylibcudf.libcudf.io.types cimport table_metadata
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
        df = cudf.DataFrame.from_pylibcudf(table)
        if index_col_count > 0:
            df = df.set_index(df._column_names[:index_col_count])
        return df

    object make_table_from_table_info_data(TableInfoData table_info, object owner):

        cdef table_metadata tbl_meta

        cdef column_name_info child_info
        for i, name in enumerate(owner._column_names):
            child_info.name = name.encode()
            tbl_meta.schema_info.push_back(child_info)
            _set_col_children_metadata(
                owner[name]._column,
                tbl_meta.schema_info[i]
            )

        index_names = None

        if (table_info.index_names.size() > 0):
            index_names = []

            for c_name in table_info.index_names:
                name = c_name.decode()
                index_names.append(name if name != "" else None)

        column_names = []

        for c_name in table_info.column_names:
            name = c_name.decode()
            column_names.append(name if name != "" else None)


        column_indicies = []

        for c_index in table_info.column_indices:
            column_indicies.append(c_index)

        data, index = data_from_table_view_indexed(
            table_info.table_view,
            owner=owner,
            column_names=column_names,
            column_indices=column_indicies,
            index_names=index_names
        )

        df = cudf.DataFrame._from_data(data, index)

        update_struct_field_names(df, tbl_meta.schema_info)

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
        all_names.extend(table._column_names)
        check_empty_name = len(all_names) == 1

        for name in table.index.names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            index_names.push_back(str.encode(name))

        for name in table._column_names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            column_names.push_back(str.encode(name))

        return TableInfoData(input_table_view, index_names, column_names)

    cdef data_from_table_view_indexed(
        table_view tv,
        object owner,
        object column_names,
        object column_indices,
        object index_names
    ):
        """
        Given a ``cudf::table_view``, constructs a Frame from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a Frame we reach inside of it and
        reach inside of each ``cudf.Column`` to make the owner of each newly
        created ``Buffer`` underneath the ``cudf.Column`` objects of the
        created Frame the respective ``Buffer`` from the relevant
        ``cudf.Column`` of the ``owner`` Frame
        """
        cdef size_type column_idx = 0
        table_owner = isinstance(owner, cudf.DataFrame)

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                column_owner = owner
                if table_owner:
                    column_owner = owner.index._columns[column_idx]
                index_columns.append(
                    ColumnBase.from_pylibcudf(
                        plc_Column.from_column_view(
                            tv.column(column_idx),
                            column_owner.to_pylibcudf(mode="read")
                        )
                    )
                )
                column_idx += 1
            
            if len(index_columns) == 1:
                index = cudf.Index._from_column(index_columns[0], name=index_names[0])
            else:
                index = cudf.MultiIndex._from_data(dict(zip(index_names, index_columns)))

        # Construct the data dict
        cdef size_type source_column_idx = 0
        data_columns = []
        for _ in column_names:
            column_owner = owner
            if table_owner:
                column_owner = owner._columns[column_indices[source_column_idx]]
            data_columns.append(
                ColumnBase.from_pylibcudf(
                    plc_Column.from_column_view(
                        tv.column(column_idx),
                        column_owner.to_pylibcudf(mode="read")
                    )
                )
            )
            column_idx += 1
            source_column_idx += 1

        return dict(zip(column_names, data_columns)), index

cdef _set_col_children_metadata(col,
                                column_name_info& col_meta):
    cdef column_name_info child_info
    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            child_info.name = name.encode()
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        for i, child_col in enumerate(col.children):
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    else:
        return

cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info
):
    for i, (name, col) in enumerate(table._data.items()):
        table._data[name] = update_column_struct_field_names(
            col, schema_info[i]
        )


cdef update_column_struct_field_names(
    col,
    column_name_info& info
):
    cdef vector[string] field_names

    if col.dtype != "object" and col.children:
        children = list(col.children)
        for i, child in enumerate(children):
            children[i] = update_column_struct_field_names(
                child,
                info.children[i]
            )
            col.set_base_children(tuple(children))

    if isinstance(col.dtype, cudf.StructDtype):
        field_names.reserve(len(col.base_children))
        for i in range(info.children.size()):
            field_names.push_back(info.children[i].name)
        col = col._rename_fields(
            field_names
        )

    return col
