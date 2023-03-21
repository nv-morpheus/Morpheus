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

#include "morpheus/messages/multi_response.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/cupy_util.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
MultiResponseMessage::MultiResponseMessage(std::shared_ptr<MessageMeta> meta,
                                           TensorIndex mess_offset,
                                           TensorIndex mess_count,
                                           std::shared_ptr<TensorMemory> memory,
                                           TensorIndex offset,
                                           TensorIndex count,
                                           std::string id_tensor_name,
                                           std::string probs_tensor_name) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count, std::move(id_tensor_name)),
  probs_tensor_name(std::move(probs_tensor_name))
{}

const TensorObject MultiResponseMessage::get_output(const std::string& name) const
{
    return get_tensor(name);
}

TensorObject MultiResponseMessage::get_output(const std::string& name)
{
    return get_tensor(name);
}

void MultiResponseMessage::set_output(const std::string& name, const TensorObject& value)
{
    set_tensor(name, value);
}

TensorObject MultiResponseMessage::get_probs_tensor() const
{
    try
    {
        return this->get_tensor(this->probs_tensor_name);
    } catch (std::runtime_error)
    {
        // Throw a better error here if we are missing the ID tensor
        throw pybind11::key_error{MORPHEUS_CONCAT_STR("Cannot get probabilities tensor. Tensor with name '"
                                                      << this->probs_tensor_name
                                                      << "' does not exist in the memory object")};
    }
}

/****** MultiResponseMessageInterfaceProxy *************************/
std::shared_ptr<MultiResponseMessage> MultiResponseMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                                               TensorIndex mess_offset,
                                                                               TensorIndex mess_count,
                                                                               std::shared_ptr<TensorMemory> memory,
                                                                               TensorIndex offset,
                                                                               TensorIndex count,
                                                                               std::string id_tensor_name,
                                                                               std::string probs_tensor_name)
{
    return std::make_shared<MultiResponseMessage>(std::move(meta),
                                                  mess_offset,
                                                  mess_count,
                                                  std::move(memory),
                                                  offset,
                                                  count,
                                                  std::move(id_tensor_name),
                                                  std::move(probs_tensor_name));
}

std::string MultiResponseMessageInterfaceProxy::probs_tensor_name_getter(MultiResponseMessage& self)
{
    return self.probs_tensor_name;
}

void MultiResponseMessageInterfaceProxy::probs_tensor_name_setter(MultiResponseMessage& self,
                                                                  std::string probs_tensor_name)
{
    self.probs_tensor_name = probs_tensor_name;
}

pybind11::object MultiResponseMessageInterfaceProxy::get_probs_tensor(MultiResponseMessage& self)
{
    return CupyUtil::tensor_to_cupy(self.get_probs_tensor());
}

}  // namespace morpheus
