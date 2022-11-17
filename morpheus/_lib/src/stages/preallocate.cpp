/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <exception>

namespace {
/**
 * @brief Performs preallocation to the underlying dataframe. These functions ensure that the MutableTableInfo object
 * has gone out of scope and thus releasing the mutex prior to the stage calling `on_next` which may block.
 *
 * @param msg
 * @param columns
 */
//@{
void preallocate(std::shared_ptr<morpheus::MessageMeta> msg,
                 const std::vector<std::tuple<std::string, morpheus::TypeId>> &columns)
{
    auto table = msg->get_mutable_info();
    std::vector<std::string> column_names;
    std::vector<morpheus::TypeId> column_types;
    for (const auto &column : columns)
    {
        column_names.push_back(std::get<0>(column));
        column_types.push_back(std::get<1>(column));
    }
    table.insert_missing_columns(column_names, column_types);
}

void preallocate(std::shared_ptr<morpheus::MultiMessage> msg,
                 const std::vector<std::tuple<std::string, morpheus::TypeId>> &columns)
{
    preallocate(msg->meta, columns);
}
//@}
}  // namespace

namespace morpheus {

template <typename MessageT>
PreallocateStage<MessageT>::PreallocateStage(const std::vector<std::tuple<std::string, std::string>> &needed_columns) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_needed_columns{needed_columns.size()}
{
    for (std::size_t i = 0; i < needed_columns.size(); ++i)
    {
        const auto dtype{DataType::from_numpy(std::get<1>(needed_columns[i]))};
        m_needed_columns[i] = std::make_tuple<>(std::get<0>(needed_columns[i]), dtype.type_id());
    }
}

template <typename MessageT>
typename PreallocateStage<MessageT>::subscribe_fn_t PreallocateStage<MessageT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                // Since the msg was just emitted from the source we shouldn't have any trouble acquiring the mutex.
                preallocate(x, m_needed_columns);
                output.on_next(std::move(x));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

template <typename MessageT>
std::shared_ptr<srf::segment::Object<PreallocateStage<MessageT>>> PreallocateStageInterfaceProxy<MessageT>::init(
    srf::segment::Builder &builder,
    const std::string &name,
    std::vector<std::tuple<std::string, std::string>> needed_columns)
{
    return builder.construct_object<PreallocateStage<MessageT>>(name, needed_columns);
}

}  // namespace morpheus
