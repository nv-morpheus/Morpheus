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
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryProbs*********************************/
/**
 * TODO(Documentation)
 */
class ResponseMemoryProbs : public ResponseMemory
{
  public:
    ResponseMemoryProbs(size_t count, TensorObject probs);

    /**
     * TODO(Documentation)
     */
    const TensorObject &get_probs() const;

    /**
     * TODO(Documentation)
     */
    void set_probs(TensorObject probs);
};

/****** ResponseMemoryProbsInterfaceProxy*******************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct ResponseMemoryProbsInterfaceProxy
{
    /**
     * @brief Create and initialize a ResponseMemoryProbs object, and return a shared pointer to the result.
     */
    static std::shared_ptr<ResponseMemoryProbs> init(cudf::size_type count, pybind11::object probs);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(ResponseMemoryProbs &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_probs(ResponseMemoryProbs &self);

    /**
     * TODO(Documentation)
     */
    static void set_probs(ResponseMemoryProbs &self, pybind11::object cupy_values);
};
#pragma GCC visibility pop
}  // namespace morpheus
