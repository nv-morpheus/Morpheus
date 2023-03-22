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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>  // for size_type

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

}  // namespace morpheus
