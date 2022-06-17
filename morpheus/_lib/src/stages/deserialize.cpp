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

#include <morpheus/stages/deserialization.hpp>

#include <pysrf/node.hpp>
#include <srf/segment/builder.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace morpheus {
// Component public implementations
// ************ DeserializationStage **************************** //
DeserializeStage::DeserializeStage(size_t batch_size) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_batch_size(batch_size)
{}

DeserializeStage::subscribe_fn_t DeserializeStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                // Make one large MultiMessage
                auto full_message = std::make_shared<MultiMessage>(x, 0, x->count());

                // Loop over the MessageMeta and create sub-batches
                for (size_t i = 0; i < x->count(); i += this->m_batch_size)
                {
                    auto next = full_message->get_slice(i, std::min(i + this->m_batch_size, x->count()));

                    output.on_next(std::move(next));
                }
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ DeserializationStageInterfaceProxy ************* //
std::shared_ptr<srf::segment::Object<DeserializeStage>> DeserializeStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name, size_t batch_size)
{
    auto stage = builder.construct_object<DeserializeStage>(name, batch_size);

    return stage;
}
}  // namespace morpheus
