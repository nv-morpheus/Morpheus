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

#include "morpheus/messages/multi_response_probs.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseProbsMessage****************************************/
MultiResponseProbsMessage::MultiResponseProbsMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                                     size_t mess_offset,
                                                     size_t mess_count,
                                                     std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                                                     size_t offset,
                                                     size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{}

const TensorObject MultiResponseProbsMessage::get_probs() const
{
    return this->get_output("probs");
}

void MultiResponseProbsMessage::set_probs(const TensorObject& probs)
{
    this->set_output("probs", probs);
}

/****** MultiResponseProbsMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
std::shared_ptr<MultiResponseProbsMessage> MultiResponseProbsMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<ResponseMemoryProbs> memory,
    cudf::size_type offset,
    cudf::size_type count)
{
    return std::make_shared<MultiResponseProbsMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::ResponseMemoryProbs> MultiResponseProbsMessageInterfaceProxy::memory(
    MultiResponseProbsMessage& self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::ResponseMemoryProbs>(self.memory) != nullptr);

    return std::static_pointer_cast<morpheus::ResponseMemoryProbs>(self.memory);
}

std::size_t MultiResponseProbsMessageInterfaceProxy::offset(MultiResponseProbsMessage& self)
{
    return self.offset;
}

std::size_t MultiResponseProbsMessageInterfaceProxy::count(MultiResponseProbsMessage& self)
{
    return self.count;
}

pybind11::object MultiResponseProbsMessageInterfaceProxy::probs(MultiResponseProbsMessage& self)
{
    // Get and convert
    auto tensor = self.get_probs();

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
