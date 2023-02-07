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

#include "morpheus/objects/table_info.hpp"

#include <ostream>
#include <string>

namespace morpheus {

/**
 * @addtogroup IO
 * @{
 * @file
 */

/**
 * @brief Serialize a dataframe to an output stream in JSON format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @return std::string
 */
std::string df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col = true);

/**
 * @brief Serialize a dataframe to an output stream in CSV format
 *
 * @param tbl : A wrapper around data in the dataframe
 * @param out_stream : Output stream to write the results to a destination
 * @param include_header : Determines whether or not to include the header
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @param flush : When `true` flush `out_stream`.
 */
void df_to_csv(const TableInfo& tbl,
               std::ostream& out_stream,
               bool include_header,
               bool include_index_col = true,
               bool flush             = false);

/**
 * @brief Serialize a dataframe into a JSON formatted string
 * @param tbl : A wrapper around data in the dataframe
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @return std::string
 *
 * Note the include_index_col is currently being ignored in both versions of `df_to_json` due to a known issue in
 * Pandas: https://github.com/pandas-dev/pandas/issues/37600
 * Requires MutableTableInfo since there is no C++ implementation of the JSON writer
 */
std::string df_to_json(MutableTableInfo& tbl, bool include_index_col = true);

/**
 * @brief Serialize a dataframe into a JSON formatted string
 * @param tbl : A wrapper around data in the dataframe
 * @param out_stream : Output stream to write the results to a destination
 * @param include_index_col : Determines whether or not to include the dataframe index
 * @param flush : When `true` flush `out_stream`.
 *
 * Requires MutableTableInfo since there is no C++ implementation of the JSON writer
 */
void df_to_json(MutableTableInfo& tbl, std::ostream& out_stream, bool include_index_col = true, bool flush = false);

/** @} */  // end of group
}  // namespace morpheus
