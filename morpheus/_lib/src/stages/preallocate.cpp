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

#include "morpheus/stages/preallocate.hpp"

#include "morpheus/utilities/type_util_detail.hpp"

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>
#include <srf/segment/builder.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace morpheus {
// Component public implementations
// ************ DeserializationStage **************************** //
PreallocateStage::PreallocateStage(const std::map<std::string, std::string> &needed_columns) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator()))
{
    for (const auto &column : needed_columns)
    {
        m_column_names.push_back(column.first);
        m_column_types.push_back(DataType::from_numpy(column.second).type_id());
    }
}

PreallocateStage::subscribe_fn_t PreallocateStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                // We want to ensure the mutable table object goes out of scope and releases the mutex prior to
                // calling on_next which may block,
                {
                    auto table = x->get_mutable_info();
                    table.insert_missing_columns(m_column_names, m_column_types);
                }
                output.on_next(std::move(x));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ DeserializationStageInterfaceProxy ************* //
std::shared_ptr<srf::segment::Object<PreallocateStage>> PreallocateStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name, const std::map<std::string, std::string> &needed_columns)
{
    auto stage = builder.construct_object<PreallocateStage>(name, needed_columns);

    return stage;
}
}  // namespace morpheus
