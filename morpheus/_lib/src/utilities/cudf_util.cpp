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

#include <morpheus/utilities/cudf_util.hpp>

#include <morpheus/objects/table_info.hpp>

#include <glog/logging.h>
#include <pybind11/gil.h>
#include <cudf/table/table.hpp>

/**
 * **************This needs to come last.********************
 * A note to posterity: We only ever want to have a single place where cudf_helpers_api.h is included in any
 * translation unit.
 */
#include "cudf_helpers_api.h"

void morpheus::load_cudf_helpers()
{
    if (import_morpheus___lib__cudf_helpers() != 0)
    {
        pybind11::error_already_set ex;

        LOG(ERROR) << "Could not load cudf_helpers library: " << ex.what();
        throw ex;
    }
}

pybind11::object morpheus::proxy_table_from_table_with_metadata(cudf::io::table_with_metadata &&table,
                                                                int index_col_count)
{
    return pybind11::reinterpret_steal<pybind11::object>(
        (PyObject *)make_table_from_table_with_metadata(std::move(table), index_col_count));
}

morpheus::TableInfo morpheus::proxy_table_info_from_table(pybind11::object table,
                                                          std::shared_ptr<const morpheus::IDataTable> idata_table)
{
    return make_table_info_from_table(table.ptr(), idata_table);
}
