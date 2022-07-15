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

#include "morpheus/objects/table_info.hpp"

#include <pybind11/pybind11.h>

namespace morpheus {
/****** Component public implementations *******************/
/****** PyDataTable****************************************/
/**
 * TODO(Documentation)
 */
struct PyDataTable : public IDataTable
{
    PyDataTable(pybind11::object &&py_table);
    ~PyDataTable();

    /**
     * TODO(Documentation)
     */
    cudf::size_type count() const override;

    /**
     * TODO(Documentation)
     */
    TableInfo get_info() const override;

    /**
     * TODO(Documentation)
     */
    const pybind11::object &get_py_object() const override;

  private:
    pybind11::object m_py_table;
};
}  // namespace morpheus
