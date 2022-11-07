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

#include "morpheus/stages/file_source.hpp"

#include "morpheus/io/deserializers.hpp"

#include <cudf/column/column.hpp>  // for column
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table.hpp>                  // for table
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/cast.h>  // for object_api::operator()
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_
#include <srf/segment/builder.hpp>

#include <algorithm>  // for find
#include <cstddef>    // for size_t
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <utility>
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ FileSourceStage ************* //
FileSourceStage::FileSourceStage(std::string filename, int repeat) :
  PythonSource(build()),
  m_filename(std::move(filename)),
  m_repeat(repeat)
{}

FileSourceStage::subscriber_fn_t FileSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        for (cudf::size_type repeat_idx = 0; repeat_idx < m_repeat; ++repeat_idx)
        {
            auto data_table     = load_table_from_file(m_filename);
            int index_col_count = get_index_col_count(data_table);

            auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);
            output.on_next(meta);
        }

        output.on_completed();
    };
}

// ************ FileSourceStageInterfaceProxy ************ //
std::shared_ptr<srf::segment::Object<FileSourceStage>> FileSourceStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name, std::string filename, int repeat)
{
    auto stage = builder.construct_object<FileSourceStage>(name, filename, repeat);

    return stage;
}
}  // namespace morpheus
