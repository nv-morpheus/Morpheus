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
#include <cudf/types.hpp>  // for size_type
#include <glog/logging.h>
#include <pybind11/pytypes.h>  // for object

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace morpheus {

/**
 * @brief Simple structure which provides a general method for holding a cudf:table_view together with index and column
 * names. Also provides slicing mechanics.
 *
 */
struct TableInfoData
{
    TableInfoData() = default;
    TableInfoData(cudf::table_view view, std::vector<std::string> indices, std::vector<std::string> columns);

    TableInfoData get_slice(std::vector<std::string> column_names = {}) const;

    TableInfoData get_slice(cudf::size_type start,
                            cudf::size_type stop,
                            std::vector<std::string> column_names = {}) const;

    cudf::table_view table_view;
    std::vector<std::string> index_names;
    std::vector<std::string> column_names;
};

/****** Component public implementations *******************/
/****** TableInfo******************************************/
struct __attribute__((visibility("default"))) TableInfoBase
{
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
     * @brief Returns a copy of the underlying cuDF DataFrame as a python object
     *
     * Note: The attribute is needed here as pybind11 requires setting symbol visibility to hidden by default.
     */
    pybind11::object copy_to_py_object() const;

    /**
     * TODO(Documentation)
     */
    const cudf::column_view &get_column(cudf::size_type idx) const;

  protected:
    TableInfoBase() = default;

    TableInfoBase(std::shared_ptr<const IDataTable> parent, TableInfoData data);

    const std::shared_ptr<const IDataTable> &get_parent() const;

    TableInfoData &get_data();
    const TableInfoData &get_data() const;

  private:
    std::shared_ptr<const IDataTable> m_parent;
    TableInfoData m_data;
};

struct __attribute__((visibility("default"))) TableInfo : public TableInfoBase
{
  public:
    TableInfo() = default;
    TableInfo(std::shared_ptr<const IDataTable> parent, std::shared_lock<std::shared_mutex> lock, TableInfoData data);

    /**
     * TODO(Documentation)
     */
    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const;

  private:
    // We use a shared_lock to allow multiple `TableInfo` to be in use at the same time.
    std::shared_lock<std::shared_mutex> m_lock;
};

struct __attribute__((visibility("default"))) MutableTableInfo : public TableInfoBase
{
  public:
    MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                     std::unique_lock<std::shared_mutex> lock,
                     TableInfoData data);

    ~MutableTableInfo();

    /**
     * TODO(Documentation)
     */
    void insert_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * TODO(Documentation)
     */
    void insert_missing_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * @brief Allows the python object to be "checked out" which gives exclusive access to the python object during the
     * lifetime of `MutableTableInfo`. Use this method when it is necessary to make changes to the python object using
     * the python API. The python object must be returned via `return_obj` before `MutableTableInfo` goes out of scope.
     *
     * @return pybind11::object
     */
    pybind11::object checkout_obj();

    /**
     * @brief Returns the checked out python object from `checkout_obj`. Each call to `checkout_obj` needs a
     * coresponding `return_obj` call.
     *
     * @param obj
     */
    void return_obj(pybind11::object &&obj);

  private:
    // We use a unique_lock here to enforce exclusive access
    std::unique_lock<std::shared_mutex> m_lock;

    mutable int m_checked_out_ref_count{-1};
};

}  // namespace morpheus
