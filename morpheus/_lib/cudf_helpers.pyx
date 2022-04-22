# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cudf import DataFrame, Series

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move
from libcpp.memory cimport make_shared, make_unique, shared_ptr, unique_ptr

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.table cimport Table, table_view_from_table
from cudf._lib.cpp.io.types cimport table_with_metadata, table_metadata
from cudf._lib.utils cimport data_from_table_view, data_from_unique_ptr

cdef extern from "morpheus/objects/table_info.hpp" namespace "morpheus" nogil:

   cdef cppclass IDataTable:
      IDataTable()

   cdef cppclass TableInfo:
      TableInfo()
      TableInfo(shared_ptr[const IDataTable] parent, table_view view, vector[string] index_names, vector[string] column_names)

      table_view get_view() const
      vector[string] get_index_names()
      vector[string] get_column_names() const

      int num_indices() const
      int num_columns() const
      int num_rows() const

cdef public api:
   Column make_column_from_view(column_view view):
      return Column.from_column_view(view, None)

   column_view make_view_from_column(Column col):
      return col.view()

   # Table make_table_from_datatable(shared_ptr[IDataTable] table, object owner):

   #    index_names = [x.decode() for x in table.get()[0].index_names]
   #    column_names = [x.decode() for x in table.get()[0].column_names]

   #    if (len(index_names) == 0):
   #       index_names = None

   #    data, index = data_from_table_view(table.get()[0].get_view(), owner, column_names=column_names, index_names=index_names)

   #    return DataFrame._from_data(data, index)

   Table make_table_from_table_with_metadata(table_with_metadata table, int index_col_count):

      index_names = None

      if (index_col_count > 0):
         index_names = [x.decode() for x in table.metadata.column_names[0:index_col_count]]

      column_names = [x.decode() for x in table.metadata.column_names[index_col_count:]]

      data, index = data_from_unique_ptr(move(table.tbl), column_names=column_names, index_names=index_names)

      return DataFrame._from_data(data, index)

   Table make_table_from_table_info(TableInfo info, object owner):

      i_names = info.get_index_names()
      c_names = info.get_column_names()

      index_names = [x.decode() for x in i_names]
      column_names = [x.decode() for x in c_names]

      data, index = data_from_table_view(info.get_view(), owner, column_names=column_names, index_names=index_names)

      return DataFrame._from_data(data, index)

   # Table make_series_from_table_info(TableInfo info, object owner):

   #    i_names = info.get_index_names()
   #    c_names = info.get_column_names()

   #    index_names = [x.decode() for x in i_names]
   #    column_names = [x.decode() for x in c_names]

   #    name = column_names[0]

   #    data, index = data_from_table_view(info.get_view(), owner, column_names=column_names, index_names=index_names)

   #    return Series._from_data(data, index, name)

   # Table make_table_from_view_and_meta(table_view view, table_metadata meta):

   #    column_names = [x.decode() for x in meta.column_names]

   #    data, index = data_from_table_view(view, None, column_names=column_names)

   #    return DataFrame._from_data(data, index)

   # TableInfo make_table_info_from_table(Table table):

   #    cdef table_view input_table_view = table_view_from_table(
   #       table
   #    )
   #    cdef table_metadata metadata_ = table_metadata()

   #    all_names = (table._index_names if table._num_indices > 0 else []) + table._column_names

   #    if len(all_names) > 0:
   #       metadata_.column_names.reserve(len(all_names))
   #       if len(all_names) == 1:
   #             if all_names[0] in (None, ''):
   #                metadata_.column_names.push_back('""'.encode())
   #             else:
   #                metadata_.column_names.push_back(
   #                   str(all_names[0]).encode()
   #                )
   #       else:
   #             for idx, col_name in enumerate(all_names):
   #                if col_name is None:
   #                   metadata_.column_names.push_back(''.encode())
   #                else:
   #                   metadata_.column_names.push_back(
   #                         str(col_name).encode()
   #                   )
   #    return TableInfo(input_table_view, metadata_, table._num_indices)

   TableInfo make_table_info_from_table(Table table, shared_ptr[const IDataTable] parent):

      cdef table_view input_table_view = table_view_from_table(table, ignore_index=False)
      cdef vector[string] index_names
      cdef vector[string] column_names

      # cuDF does a weird check where if there is only one name in both index and columns, and that column is empty or
      # None, then change it to '""'. Not sure what this is used for
      check_empty_name = (table._num_indices + table._num_columns) == 1

      for name in table._index_names:
         if (check_empty_name and name in (None, '')):
            name = '""'
         elif (name is None):
            name = ""

         index_names.push_back(name.encode())

      for name in table._column_names:
         if (check_empty_name and name in (None, '')):
            name = '""'
         elif (name is None):
            name = ""

         column_names.push_back(name.encode())

      return TableInfo(parent, input_table_view, index_names, column_names)
