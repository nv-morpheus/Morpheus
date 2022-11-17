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
    pybind11::object get_py_table() const;

    /**
     * @brief Get messages count
     * 
     * @return size_t 
     */
    size_t count() const;

    /**
     * @brief Get the info object
     * 
     * @return TableInfo 
     */
    TableInfo get_info() const;

    /**
     * @brief Create MessageMeta cpp object from a python object
     * 
     * @param data_table 
     * @return std::shared_ptr<MessageMeta> 
     */
    static std::shared_ptr<MessageMeta> create_from_python(pybind11::object&& data_table);

    /**
     * @brief Create MessageMeta cpp object from a cpp object
     * 
     * @param data_table 
     * @param index_col_count 
     * @return std::shared_ptr<MessageMeta> 
     */
    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        int index_col_count = 0);

  private:
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
     * @brief Get the data frame object
     * 
     * @param self 
     * @return pybind11::object 
     */
    static pybind11::object get_data_frame(MessageMeta& self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
