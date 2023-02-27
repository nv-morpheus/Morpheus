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

#include "morpheus/types.hpp"  // for TensorIndex & tensor_map_t

#include <cstddef>  // for size_t
#include <string>
#include <utility>  // for pair
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Container for holding a collection of named `TensorObject`s in a `std::map` keyed by name.
 * Base class for `InferenceMemory` & `ResponseMemory`
 *
 */
class TensorMemory
{
  public:
    /**
     * @brief Construct a new Tensor Memory object
     *
     * @param count
     */
    TensorMemory(size_t count);

    /**
     * @brief Construct a new Tensor Memory object
     *
     * @param count
     * @param tensors
     */
    TensorMemory(size_t count, tensor_map_t&& tensors);
    virtual ~TensorMemory() = default;

    size_t count{0};
    tensor_map_t tensors;

    /**
     * @brief Verify whether the specified tensor name is present in the tensor memory
     *
     * @param name
     * @return true
     * @return false
     */
    bool has_tensor(const std::string& name) const;

    /**
     * @brief Copy tensor ranges
     *
     * @param ranges
     * @param num_selected_rows
     * @return tensor_map_t
     */
    tensor_map_t copy_tensor_ranges(const std::vector<std::pair<TensorIndex, TensorIndex>>& ranges,
                                    size_t num_selected_rows) const;
};

/** @} */  // end of group
}  // namespace morpheus
