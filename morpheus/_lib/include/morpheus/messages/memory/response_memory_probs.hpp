/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryProbs*********************************/

/**
 * @addtogroup messages
 * @{
 * @file
*/

/**
 * @brief Output memory block containing the inference response probabilities.
 * 
 */
class ResponseMemoryProbs : public ResponseMemory
{
  public:
    /**
     * @brief Construct a new Response Memory Probs object
     * 
     * @param count 
     * @param probs 
     */
    ResponseMemoryProbs(size_t count, TensorObject probs);
    /**
     * @brief Construct a new Response Memory Probs object
     * 
     * @param count 
     * @param tensors 
     */
    ResponseMemoryProbs(size_t count, tensor_map_t &&tensors);

    /**
     * @brief Returns the tensor named 'probs', throws a `std::runtime_error` if it does not exist
     *
     * @return const TensorObject&
     */
    const TensorObject &get_probs() const;

    /**
     * @brief Update the tensor named 'probs'
     *
     * @param probs
     */
    void set_probs(TensorObject probs);
};

/****** ResponseMemoryProbsInterfaceProxy*******************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct ResponseMemoryProbsInterfaceProxy
{
    /**
     * @brief Create and initialize a ResponseMemoryProbs object, and return a shared pointer to the result
     * 
     * @param count 
     * @param probs 
     * @return std::shared_ptr<ResponseMemoryProbs> 
     */
    static std::shared_ptr<ResponseMemoryProbs> init(cudf::size_type count, pybind11::object probs);

    /**
     * @brief Get messages count in the response memory probs object
     * 
     * @param self 
     * @return std::size_t 
     */
    static std::size_t count(ResponseMemoryProbs &self);

    /**
     * @brief Get the response memory probs object
     * 
     * @param self 
     * @return pybind11::object 
     */
    static pybind11::object get_probs(ResponseMemoryProbs &self);

    /**
     * @brief Set the response memory probs object
     * 
     * @param self 
     * @param cupy_values 
     */
    static void set_probs(ResponseMemoryProbs &self, pybind11::object cupy_values);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
