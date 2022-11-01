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
    // /**
    //  * TODO(Documentation)
    //  */
    // const pybind11::object &get_parent_table() const;

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

    // cudf::table_view m_table_view;
    // std::vector<std::string> m_column_names;
    // std::vector<std::string> m_index_names;
};

struct __attribute__((visibility("default"))) TableInfo : public TableInfoBase
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

// class PyObjectLocked : public pybind11::object
// {
//   public:
//     // // Returns const ref. Used by object_api. Should not be used directly. Requires the GIL
//     // operator const pybind11::handle &() const &;

//     // // Necessary to implement the object_api interface
//     // PyObject *ptr() const;
//     PyObjectLocked(pybind11::object &&obj);
//     PyObjectLocked(const PyObjectLocked &other) = delete;

//     PyObjectLocked &operator=(const PyObjectLocked &other) = delete;

//     PyObjectLocked &operator=(const pybind11::object &obj);

//     operator pybind11::object() const & = delete;
// };

struct __attribute__((visibility("default"))) MutableTableInfo : public TableInfoBase
{
  public:
    MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                     std::unique_lock<std::shared_mutex> lock,
                     TableInfoData data) :
      TableInfoBase(parent, std::move(data)),
      m_lock(std::move(lock))
    {}

    ~MutableTableInfo()
    {
        if (m_checked_out_ref_count >= 0)
        {
            LOG(FATAL) << "Checked out python object was not returned before MutableTableInfo went out of scope";
        }
    }

    /**
     * TODO(Documentation)
     */
    void insert_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    /**
     * TODO(Documentation)
     */
    void insert_missing_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types);

    // PyObjectLocked py_obj() const{
    //     pybind11::object obj;

    //     obj.ref_count()
    // }

    pybind11::object checkout_obj();

    void return_obj(pybind11::object &&obj);

  private:
    std::unique_lock<std::shared_mutex> m_lock;

    mutable int m_checked_out_ref_count{-1};
};

}  // namespace morpheus
