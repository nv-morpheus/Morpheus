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
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace morpheus {

struct TableInfoData
{
    TableInfoData() = default;
    TableInfoData(cudf::table_view view, std::vector<std::string> indices, std::vector<std::string> columns);

    cudf::table_view table_view;
    std::vector<std::string> index_names;
    std::vector<std::string> column_names;
};

/****** Component public implementations *******************/
/****** TableInfo******************************************/
struct TableInfoBase
{
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
     * @brief Returns the underlying cuDF DataFrame as a python object
     *
     * Note: The attribute is needed here as pybind11 requires setting symbol visibility to hidden by default.
     */
    pybind11::object __attribute__((visibility("default"))) as_py_object() const;

    /**
     * TODO(Documentation)
     */
    const cudf::column_view &get_column(cudf::size_type idx) const;

  protected:
    TableInfoBase() = default;

    TableInfoBase(std::shared_ptr<const IDataTable> parent, TableInfoData data);

    std::shared_ptr<const IDataTable> m_parent;
    cudf::table_view m_table_view;
    std::vector<std::string> m_column_names;
    std::vector<std::string> m_index_names;
};

struct TableInfo : public TableInfoBase
{
  public:
    TableInfo() = default;
    TableInfo(std::shared_ptr<const IDataTable> parent, std::shared_lock<std::shared_mutex> lock, TableInfoData data) :
      TableInfoBase(parent, std::move(data)),
      m_lock(std::move(lock))
    {}

    /**
     * TODO(Documentation)
     */
    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const;

  private:
    std::shared_lock<std::shared_mutex> m_lock;
};

struct MutableTableInfo : public TableInfoBase
{
  public:
    MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                     std::unique_lock<std::shared_mutex> lock,
                     TableInfoData data) :
      TableInfoBase(parent, std::move(data)),
      m_lock(std::move(lock))
    {}
    /**
     * TODO(Documentation)
     */
    void insert_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * TODO(Documentation)
     */
    void insert_missing_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

  private:
    std::unique_lock<std::shared_mutex> m_lock;
};
}  // namespace morpheus
