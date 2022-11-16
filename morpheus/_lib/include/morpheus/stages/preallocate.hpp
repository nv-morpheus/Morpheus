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

#pragma once

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/utilities/type_util_detail.hpp"  // for TypeId

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {
/**
 * @brief Performs preallocation to the underlying dataframe. These functions ensure that the MutableTableInfo object
 * has gone out of scope and thus releasing the mutex prior to the stage calling `on_next` which may block.
 *
 * @param msg
 * @param column_names
 * @param column_types
 */
//@{
void preallocate(std::shared_ptr<morpheus::MessageMeta> msg,
                 const std::vector<std::string> &column_names,
                 const std::vector<morpheus::TypeId> &column_types)
{
    auto table = msg->get_mutable_info();
    table.insert_missing_columns(column_names, column_types);
}

void preallocate(std::shared_ptr<morpheus::MultiMessage> msg,
                 const std::vector<std::string> &column_names,
                 const std::vector<morpheus::TypeId> &column_types)
{
    preallocate(msg->meta, column_names, column_types);
}
//@}
}  // namespace

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations *******************/
/****** PreallocateStage ********************************/
/* Preallocates new columns into the underlying dataframe. This stage supports both MessageMeta & subclasses of
 * MultiMessage. In the Python bindings the stage is bound as `PreallocateMessageMetaStage` and
 * `PreallocateMultiMessageStage`
 */
template <typename MessageT>
class PreallocateStage : public srf::pysrf::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreallocateStage(const std::map<std::string, std::string> &needed_columns) :
      base_t(base_t::op_factory_from_sub_fn(build_operator()))
    {
        for (const auto &column : needed_columns)
        {
            m_column_names.push_back(column.first);
            m_column_types.push_back(DataType::from_numpy(column.second).type_id());
        }
    }

  private:
    subscribe_fn_t build_operator()
    {
        return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
            return input.subscribe(rxcpp::make_observer<sink_type_t>(
                [this, &output](sink_type_t x) {
                    // Since the msg was just emitted from the source we shouldn't have any trouble acquiring the mutex.
                    preallocate(x, m_column_names, m_column_types);
                    output.on_next(std::move(x));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::vector<std::string> m_column_names;
    std::vector<TypeId> m_column_types;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
template <typename MessageT>
struct PreallocateStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<PreallocateStage<MessageT>>> init(
        srf::segment::Builder &builder, const std::string &name, std::map<std::string, std::string> needed_columns)
    {
        auto stage = builder.construct_object<PreallocateStage<MessageT>>(name, needed_columns);

        return stage;
    }
};
#pragma GCC visibility pop
}  // namespace morpheus
