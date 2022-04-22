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

#include <morpheus/stages/deserialization.hpp>

#include <pyneo/node.hpp>
#include <neo/core/segment.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace morpheus {
    // Component public implementations
    // ************ DeserializationStage **************************** //
    DeserializeStage::DeserializeStage(const neo::Segment &parent, const std::string &name, size_t batch_size) :
            neo::SegmentObject(parent, name),
            PythonNode(parent, name, build_operator()),
            m_batch_size(batch_size) {}

    DeserializeStage::operator_fn_t DeserializeStage::build_operator() {
        return [this](neo::Observable<reader_type_t> &input, neo::Subscriber<writer_type_t> &output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                    [this, &output](reader_type_t &&x) {
                        // Make one large MultiMessage
                        auto full_message = std::make_shared<MultiMessage>(x, 0, x->count());

                        // Loop over the MessageMeta and create sub-batches
                        for (size_t i = 0; i < x->count(); i += this->m_batch_size) {
                            auto next = full_message->get_slice(i, std::min(i + this->m_batch_size, x->count()));

                            output.on_next(std::move(next));
                        }
                    },
                    [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                    [&]() { output.on_completed(); }));
        };
    }

    // ************ DeserializationStageInterfaceProxy ************* //
    std::shared_ptr<DeserializeStage>
    DeserializeStageInterfaceProxy::init(neo::Segment &parent, const std::string &name, size_t batch_size) {
        auto stage = std::make_shared<DeserializeStage>(parent, name, batch_size);

        parent.register_node<DeserializeStage>(stage);

        return stage;
    }
}
