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

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

namespace morpheus {
class TableInfo;

/****** Component public implementations *******************/
/****** IDataTable******************************************/

/**
 * @brief Owning object which owns a unique_ptr<cudf::table>, table_metadata, and index information
 * Why this doesnt exist in cudf is beyond me
 */
struct IDataTable : public std::enable_shared_from_this<IDataTable>
{
    IDataTable() = default;

    /**
     * TODO(Documentation)
     */
    virtual cudf::size_type count() const = 0;

    /**
     * TODO(Documentation)
     */
    virtual TableInfo get_info() const = 0;

    /**
     * TODO(Documentation)
     */
    virtual const pybind11::object &get_py_object() const = 0;
};
}  // namespace morpheus
