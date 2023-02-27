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

#pragma once

#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/types.hpp"  // for tensor_map_t

#include <cstddef>
#include <string>

namespace morpheus {
/**
 * @addtogroup messages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic data as a `TensorObject`s
 *
 */
class InferenceMemory : public TensorMemory
{
  public:
    /**
     * @brief Construct a new Inference Memory object
     *
     * @param count
     */
    InferenceMemory(size_t count);
    /**
     * @brief Construct a new Inference Memory object
     *
     * @param count
     * @param tensors
     */
    InferenceMemory(size_t count, tensor_map_t&& tensors);

    /**
     * @brief Checks if a tensor named `name` exists in `tensors`
     *
     * @param name
     * @return true
     * @return false
     */
    bool has_input(const std::string& name) const;
};

/****** InferenceMemoryInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct InferenceMemoryInterfaceProxy
{
    /**
     * @brief Get the count object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t get_count(InferenceMemory& self);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
