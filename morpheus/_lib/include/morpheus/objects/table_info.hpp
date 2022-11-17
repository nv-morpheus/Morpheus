/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "morpheus/objects/data_table.hpp"
#include "morpheus/utilities/type_util_detail.hpp"

#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>      // for size_type
#include <pybind11/pytypes.h>  // for object

#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TableInfo******************************************/

/**
 * @addtogroup objects
 * @{
 * @file
*/

/**
 * A wrapper class around a data table, which in practice a cudf dataframe. It gives the flexibility to perform operations 
 * on a data table
*/
struct TableInfo
{
    TableInfo() = default;

    TableInfo(std::shared_ptr<const IDataTable> parent,
              cudf::table_view view,
              std::vector<std::string> index_names,
              std::vector<std::string> column_names);

    /**
     * @brief Get reference of underlying cuDF DataFrame as a python object
     * 
     * @return pybind11::object
     */
    const pybind11::object &get_parent_table() const;

    /**
     * @brief Get reference of a cudf table view
     * 
     * @return cudf::table_view
     */
    const cudf::table_view &get_view() const;

    /**
     * @brief Get index names of a data table
     * 
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_index_names() const;

    /**
     * @brief Get column names of a data table
     * 
     * @return std::vector<std::string>
     */
    std::vector<std::string> get_column_names() const;

    /**
     * @brief Get size of a index names in a data table
     * 
     * @return cudf::size_type
     */
    cudf::size_type num_indices() const;

    /**
     * @brief Get columns count in a data table
     * 
     * @return cudf::size_type
     */
    cudf::size_type num_columns() const;

    /**
     * @brief Get rows count in a data table
     * 
     * @return cudf::size_type
     */
    cudf::size_type num_rows() const;

    /**
     * @brief Returns the underlying cuDF DataFrame as a python object
     *
     * Note: The attribute is needed here as pybind11 requires setting symbol visibility to hidden by default
     */
    pybind11::object __attribute__((visibility("default"))) as_py_object() const;

    /**
     * @brief Insert new columns to a data table with the value zero for each row
     * 
     * @param column_names : Names of the columns to be added to a table
     * @param column_types : Column data types
     */
    void insert_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * @brief Insert missing columns to a data table with the value zero for each row
     * 
     * @param column_names : Names of the columns to be added to a table
     * @param column_types : Column data types
     */
    void insert_missing_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * @brief Returns a reference to the view of the specified column
     * 
     * @throws std::out_of_range
     * If `column_index` is out of the range [0, num_columns)
     * 
     * @param idx : The index of the desired column
     * @return cudf::column_view : A reference to the desired column
     */
    const cudf::column_view &get_column(cudf::size_type idx) const;

    /**
     * @brief Get slice of a data table info based on the start and stop offset address
     * 
     * @param start : Start offset address
     * @param stop : Stop offset address
     * @param column_names : Columns of interest
     * @return TableInfo
     */
    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const;

  private:
    std::shared_ptr<const IDataTable> m_parent;
    cudf::table_view m_table_view;
    std::vector<std::string> m_column_names;
    std::vector<std::string> m_index_names;
};

/** @} */  // end of group
}  // namespace morpheus
