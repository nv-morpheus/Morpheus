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

#include "morpheus/messages/memory/inference_memory.hpp"

#include <cudf/io/types.hpp>

#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemory****************************************/
InferenceMemory::InferenceMemory(size_t count) : count(count) {}

bool InferenceMemory::has_input(const std::string& name) const
{
    return this->inputs.find(name) != this->inputs.end();
}

/****** InferenceMemoryInterfaceProxy *************************/
std::size_t InferenceMemoryInterfaceProxy::get_count(InferenceMemory& self)
{
    return self.count;
}
}  // namespace morpheus
