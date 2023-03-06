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

#include "morpheus/messages/memory/inference_memory.hpp"

// for TensorObject
#include "morpheus/objects/tensor_object.hpp"  // IWYU pragma: keep

#include <string>
#include <utility>  // for move

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemory****************************************/
InferenceMemory::InferenceMemory(size_t count) : TensorMemory(count) {}
InferenceMemory::InferenceMemory(size_t count, TensorMapType&& tensors) : TensorMemory(count, std::move(tensors)) {}

bool InferenceMemory::has_input(const std::string& name) const
{
    return this->has_tensor(name);
}

/****** InferenceMemoryInterfaceProxy *************************/
std::size_t InferenceMemoryInterfaceProxy::get_count(InferenceMemory& self)
{
    return self.count;
}
}  // namespace morpheus
