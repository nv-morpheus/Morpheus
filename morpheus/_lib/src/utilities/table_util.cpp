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

#include <morpheus/utilities/table_util.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>

#include <glog/logging.h>

#include <filesystem>
#include <memory>

namespace fs = std::filesystem;
namespace py = pybind11;

cudf::io::table_with_metadata morpheus::CuDFTableUtil::load_table(const std::string &filename) {
    auto file_path = fs::path(filename);

    if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines") {
        // First, load the file into json
        auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(true);

        return cudf::io::read_json(options.build());
    } else if (file_path.extension() == ".csv") {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});

        return cudf::io::read_csv(options.build());
    } else {
        LOG(FATAL) << "Unknown extension for file: " << filename;
        throw std::runtime_error("Unknown extension");
    }
}