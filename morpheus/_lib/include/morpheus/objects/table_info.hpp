/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/objects/data_table.hpp>

#include <cudf/table/table_view.hpp>
#include <neo/utils/type_utils.hpp>

#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TableInfo******************************************/
struct TableInfo
{
    TableInfo() = default;

    TableInfo(std::shared_ptr<const IDataTable> parent,
              cudf::table_view view,
              std::vector<std::string> index_names,
              std::vector<std::string> column_names);

    /**
     * TODO(Documentation)
     */
    const pybind11::object &get_parent_table() const;

    /**
     * TODO(Documentation)
     */
    const cudf::table_view &get_view() const;

    /**
     * TODO(Documentation)
     */
    std::vector<std::string> get_index_names() const;

    /**
     * TODO(Documentation)
     */
    std::vector<std::string> get_column_names() const;

    /**
     * TODO(Documentation)
     */
    cudf::size_type num_indices() const;

    /**
     * TODO(Documentation)
     */
    cudf::size_type num_columns() const;

    /**
     * TODO(Documentation)
     */
    cudf::size_type num_rows() const;

    /**
     * TODO(Documentation)
     */
    pybind11::object as_py_object() const;

    /**
     * TODO(Documentation)
     */
    void insert_columns(const std::vector<std::string> &column_names, const std::vector<neo::TypeId> &column_types);

    /**
     * TODO(Documentation)
     */
    void insert_missing_columns(const std::vector<std::string> &column_names,
                                const std::vector<neo::TypeId> &column_types);

    /**
     * TODO(Documentation)
     */
    const cudf::column_view &get_column(cudf::size_type idx) const;

    /**
     * TODO(Documentation)
     */
    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const;

  private:
    std::shared_ptr<const IDataTable> m_parent;
    cudf::table_view m_table_view;
    std::vector<std::string> m_column_names;
    std::vector<std::string> m_index_names;
};
}  // namespace morpheus
