/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>

#include <string>

namespace morpheus {
#pragma GCC visibility push(default)
/**
 * @addtogroup IO
 * @{
 * @file
 */

/**
 * @brief Loads a cudf table from either CSV or JSON file
 *
 * @param filename : Name of the file that should be loaded into a table
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata load_table_from_file(const std::string& filename);

/**
 * @brief Loads a cudf table from a JSON source, replacing any escape characters in the source data that cudf can't
 * handle
 *
 * @param json_options : JSON file reader options
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata load_json_table(cudf::io::json_reader_options&& json_options);

/**
 * @brief Returns the number of index columns in `data_table`, in practice this will be a `0` or `1`
 * If `data_table` contains a column named "Unnamed: 0" it will be renamed to ""
 *
 * @param data_table : Table which contains the data and it's metadata
 * @return int
 */
int get_index_col_count(cudf::io::table_with_metadata& data_table);

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
