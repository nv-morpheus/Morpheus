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
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory::tensor_map_t
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <cstdint>  // for int32_t
#include <memory>
#include <ostream>  // needed for logging
#include <string>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** <MultiInferenceMessage>****************************************/
MultiInferenceMessage::MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                             std::size_t mess_offset,
                                             std::size_t mess_count,
                                             std::shared_ptr<morpheus::InferenceMemory> memory,
                                             std::size_t offset,
                                             std::size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{}

const TensorObject MultiInferenceMessage::get_input(const std::string &name) const
{
    return get_tensor(name);
}

TensorObject MultiInferenceMessage::get_input(const std::string &name)
{
    return get_tensor(name);
}

void MultiInferenceMessage::set_input(const std::string &name, const TensorObject &value)
{
    set_tensor(name, value);
}

/****** <MultiInferenceMessage>InterfaceProxy *************************/
std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<InferenceMemory> memory,
    cudf::size_type offset,
    cudf::size_type count)
{
    return std::make_shared<MultiInferenceMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::InferenceMemory> MultiInferenceMessageInterfaceProxy::memory(MultiInferenceMessage &self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::InferenceMemory>(self.memory) != nullptr);
    return std::static_pointer_cast<morpheus::InferenceMemory>(self.memory);
}

std::size_t MultiInferenceMessageInterfaceProxy::offset(MultiInferenceMessage &self)
{
    return self.offset;
}

std::size_t MultiInferenceMessageInterfaceProxy::count(MultiInferenceMessage &self)
{
    return self.count;
}

pybind11::object MultiInferenceMessageInterfaceProxy::get_input(MultiInferenceMessage &self, const std::string &name)
{
    const auto &py_tensor = CupyUtil::tensor_to_cupy(self.get_input(name));
    return py_tensor;
}

std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::get_slice(MultiInferenceMessage &self,
                                                                                      std::size_t start,
                                                                                      std::size_t stop)
{
    return self.get_slice(start, stop);
}
}  // namespace morpheus
