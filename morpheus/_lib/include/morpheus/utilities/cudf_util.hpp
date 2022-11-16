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

#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>  // for shared_ptr

namespace morpheus {
/****** Component public free function implementations******/

/**
 * @addtogroup utilities
 * @{
 * @file
*/

/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
void load_cudf_helpers();
#pragma GCC visibility pop

/**
 * @brief These proxy functions allow us to have a shared set of cudf_helpers interfaces declarations, which proxy
 * the actual generated cython calls. The cython implementation in 'cudf_helpers_api.h' can only appear in the
 * translation unit for the pybind module declaration.
 */
pybind11::object proxy_table_from_table_with_metadata(cudf::io::table_with_metadata &&, int);
TableInfo proxy_table_info_from_table(pybind11::object table, std::shared_ptr<morpheus::IDataTable const> idata_table);

/**
 * @brief cudf_helper stubs -- currently not used anywhere
 */
pybind11::object /*PyColumn*/ proxy_column_from_view(cudf::column_view view);
cudf::column_view proxy_view_from_column(pybind11::object *column /*PyColumn**/);
pybind11::object /*PyTable*/ proxy_table_from_table_info(morpheus::TableInfo table_info, pybind11::object *object);
pybind11::object /*PyTable*/ proxy_series_from_table_info(morpheus::TableInfo table_info, pybind11::object *object);

/** @} */  // end of group
}  // namespace morpheus
