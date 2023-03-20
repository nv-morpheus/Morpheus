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

#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/meta.hpp"

#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseProbsMessage****************************************/
MultiResponseProbsMessage::MultiResponseProbsMessage(std::shared_ptr<MessageMeta> meta,
                                                     TensorIndex mess_offset,
                                                     TensorIndex mess_count,
                                                     std::shared_ptr<TensorMemory> memory,
                                                     TensorIndex offset,
                                                     TensorIndex count,
                                                     std::string id_tensor_name,
                                                     std::string probs_tensor_name) :
  DerivedMultiMessage(
      meta, mess_offset, mess_count, memory, offset, count, std::move(id_tensor_name), std::move(probs_tensor_name))
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
    TensorIndex mess_offset,
    TensorIndex mess_count,
    std::shared_ptr<TensorMemory> memory,
    TensorIndex offset,
    TensorIndex count,
    std::string id_tensor_name,
    std::string probs_tensor_name)
{
    return std::make_shared<MultiResponseProbsMessage>(std::move(meta),
                                                       mess_offset,
                                                       mess_count,
                                                       std::move(memory),
                                                       offset,
                                                       count,
                                                       std::move(id_tensor_name),
                                                       std::move(probs_tensor_name));
}

pybind11::object MultiResponseProbsMessageInterfaceProxy::probs(MultiResponseProbsMessage& self)
{
    return get_tensor_property(self, "probs");
}

}  // namespace morpheus
