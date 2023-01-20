/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/data_table.hpp"  // for IDataTable
#include "morpheus/objects/table_info.hpp"

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>  // for size_type
#include <pybind11/pytypes.h>

#include <cstddef>  // for size_t
#include <memory>
#include <string>

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations ******************/
/****** MessageMeta****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Container for class holding a data table, in practice a cudf DataFrame, with the ability to return both
 * Python and C++ representations of the table
 *
 */
class MessageMeta
{
  public:
    /**
     * @brief Get the py table object
     *
     * @return pybind11::object
     */
    size_t count() const;

    /**
     * @brief Get the info object
     *
     * @return TableInfo
     */
    virtual TableInfo get_info() const;

    /**
     * TODO(Documentation)
     */
    virtual MutableTableInfo get_mutable_info() const;

    /**
     * @brief Create MessageMeta cpp object from a python object
     *
     * @param data_table
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> create_from_python(pybind11::object&& data_table);

    /**
     * @brief Create MessageMeta cpp object from a cpp object, used internally by `create_from_cpp`
     *
     * @param data_table
     * @param index_col_count
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        int index_col_count = 0);

  protected:
    MessageMeta(std::shared_ptr<IDataTable> data);

    /**
     * @brief Create MessageMeta python object from a cpp object
     *
     * @param table
     * @param index_col_count
     * @return pybind11::object
     */
    static pybind11::object cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count = 0);

    std::shared_ptr<IDataTable> m_data;
};

/**
 * @brief Operates similarly to MessageMeta, except it applies a filter on the columns and rows. Used by Serialization
 * to filter columns without copying the entire DataFrame
 *
 */
class SlicedMessageMeta : public MessageMeta
{
  public:
    SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                      cudf::size_type start            = 0,
                      cudf::size_type stop             = -1,
                      std::vector<std::string> columns = {});

    TableInfo get_info() const override;

    MutableTableInfo get_mutable_info() const override;

  private:
    cudf::size_type m_start{0};
    cudf::size_type m_stop{-1};
    std::vector<std::string> m_column_names;
};

/****** Python Interface **************************/
class MutableTableCtxMgr : public std::enable_shared_from_this<MutableTableCtxMgr>
{
  public:
    MutableTableCtxMgr(MessageMeta& meta_msg);
    pybind11::object enter();
    void exit(const pybind11::object& type, const pybind11::object& value, const pybind11::object& traceback);

    // Throws a useful exception when a user attempts to use this object as if it were the dataframe itself
    void throw_usage_error(pybind11::args args, const pybind11::kwargs& kwargs);

  private:
    MessageMeta& m_meta_msg;
    std::unique_ptr<MutableTableInfo> m_table;
    std::unique_ptr<pybind11::object> m_py_table;
};

/****** MessageMetaInterfaceProxy**************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MessageMetaInterfaceProxy
{
    /**
     * @brief Initialize MessageMeta cpp object with the given filename
     *
     * @param filename : Filename for loading the data on to MessageMeta
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> init_cpp(const std::string& filename);

    /**
     * @brief Initialize MessageMeta cpp object with a given dataframe and returns shared pointer as the result
     *
     * @param data_frame : Dataframe that contains the data
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<MessageMeta> init_python(pybind11::object&& data_frame);

    /**
     * @brief Get messages count
     *
     * @param self
     * @return cudf::size_type
     */
    static cudf::size_type count(MessageMeta& self);

    /**
     * @brief Get a copy of the data frame object as a python object
     *
     * @param self The MessageMeta instance
     * @return pybind11::object A `DataFrame` object
     */
    static pybind11::object get_data_frame(MessageMeta& self);
    static pybind11::object df_property(MessageMeta& self);

    static MutableTableCtxMgr mutable_dataframe(MessageMeta& self);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
