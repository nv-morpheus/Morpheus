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

#include <morpheus/objects/table_info.hpp>

#include <pybind11/pybind11.h>
#include <cudf/io/types.hpp>

#include <memory>
#include <string>

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations ******************/
/****** MessageMeta****************************************/
/**
 * TODO(Documentation)
 */
class MessageMeta
{
  public:
    /**
     * TODO(Documentation)
     */
    pybind11::object get_py_table() const;

    /**
     * TODO(Documentation)
     */
    size_t count() const;

    /**
     * @brief
     *
     */
    TableInfo get_info() const;

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MessageMeta> create_from_python(pybind11::object&& data_table);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        int index_col_count = 0);

  private:
    MessageMeta(std::shared_ptr<IDataTable> data);

    /**
     * TODO(Documentation)
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
     * TODO(Documentation)
     */
    static std::shared_ptr<MessageMeta> init_cpp(const std::string& filename);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MessageMeta> init_python(pybind11::object&& data_frame);

    /**
     * TODO(Documentation)
     */
    static cudf::size_type count(MessageMeta& self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_data_frame(MessageMeta& self);
};
#pragma GCC visibility pop
}  // namespace morpheus
