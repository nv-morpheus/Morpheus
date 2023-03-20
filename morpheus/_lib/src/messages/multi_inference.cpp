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

#include "morpheus/messages/multi_inference.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"

#include <memory>
#include <string>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** <MultiInferenceMessage>****************************************/
MultiInferenceMessage::MultiInferenceMessage(std::shared_ptr<MessageMeta> meta,
                                             TensorIndex mess_offset,
                                             TensorIndex mess_count,
                                             std::shared_ptr<InferenceMemory> memory,
                                             TensorIndex offset,
                                             TensorIndex count) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{}

const TensorObject MultiInferenceMessage::get_input(const std::string& name) const
{
    return get_tensor(name);
}

TensorObject MultiInferenceMessage::get_input(const std::string& name)
{
    return get_tensor(name);
}

void MultiInferenceMessage::set_input(const std::string& name, const TensorObject& value)
{
    set_tensor(name, value);
}

/****** <MultiInferenceMessage>InterfaceProxy *************************/
std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    TensorIndex mess_offset,
    TensorIndex mess_count,
    std::shared_ptr<InferenceMemory> memory,
    TensorIndex offset,
    TensorIndex count)
{
    return std::make_shared<MultiInferenceMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

}  // namespace morpheus
