/*
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

#include "morpheus/messages/multi_inference_nlp.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi_inference.hpp"

#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiInferenceNLPMessage****************************************/
MultiInferenceNLPMessage::MultiInferenceNLPMessage(std::shared_ptr<MessageMeta> meta,
                                                   TensorIndex mess_offset,
                                                   TensorIndex mess_count,
                                                   std::shared_ptr<TensorMemory> memory,
                                                   TensorIndex offset,
                                                   TensorIndex count,
                                                   std::string id_tensor_name) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count, std::move(id_tensor_name))
{}

const TensorObject MultiInferenceNLPMessage::get_input_ids() const
{
    return this->get_input("input_ids");
}

void MultiInferenceNLPMessage::set_input_ids(const TensorObject& input_ids)
{
    this->set_input("input_ids", input_ids);
}

const TensorObject MultiInferenceNLPMessage::get_input_mask() const
{
    return this->get_input("input_mask");
}

void MultiInferenceNLPMessage::set_input_mask(const TensorObject& input_mask)
{
    this->set_input("input_mask", input_mask);
}

const TensorObject MultiInferenceNLPMessage::get_seq_ids() const
{
    return this->get_input("seq_ids");
}

void MultiInferenceNLPMessage::set_seq_ids(const TensorObject& seq_ids)
{
    this->set_input("seq_ids", seq_ids);
}

/****** MultiInferenceNLPMessageInterfaceProxy *************************/
std::shared_ptr<MultiInferenceNLPMessage> MultiInferenceNLPMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    TensorIndex mess_offset,
    TensorIndex mess_count,
    std::shared_ptr<TensorMemory> memory,
    TensorIndex offset,
    TensorIndex count,
    std::string id_tensor_name)
{
    return std::make_shared<MultiInferenceNLPMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count, std::move(id_tensor_name));
}

pybind11::object MultiInferenceNLPMessageInterfaceProxy::input_ids(MultiInferenceNLPMessage& self)
{
    return get_tensor_property(self, "input_ids");
}

pybind11::object MultiInferenceNLPMessageInterfaceProxy::input_mask(MultiInferenceNLPMessage& self)
{
    return get_tensor_property(self, "input_mask");
}

pybind11::object MultiInferenceNLPMessageInterfaceProxy::seq_ids(MultiInferenceNLPMessage& self)
{
    return get_tensor_property(self, "seq_ids");
}

}  // namespace morpheus
