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

#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/utilities/cupy_util.hpp"    // for CupyUtil

#include <pybind11/pytypes.h>  // for object

#include <cstddef>  // for size_t
#include <map>
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

#pragma GCC visibility push(default)
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
    TensorMemory(size_t count, CupyUtil::tensor_map_t&& tensors);
    virtual ~TensorMemory() = default;

    size_t count{0};
    CupyUtil::tensor_map_t tensors;

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
     * @return CupyUtil::tensor_map_t
     */
    CupyUtil::tensor_map_t copy_tensor_ranges(const std::vector<std::pair<TensorIndex, TensorIndex>>& ranges,
                                              size_t num_selected_rows) const;
};

/****** TensorMemoryInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct TensorMemoryInterfaceProxy
{
    /**
     * @brief Create and initialize a TensorMemory object, and return a shared pointer to the result. Each array in
     * `tensors` should be of length `count`.
     *
     * @param count : Lenght of each array in `tensors`
     * @param tensors : Map of string on to cupy arrays
     * @return std::shared_ptr<TensorMemory>
     */
    static std::shared_ptr<TensorMemory> init(std::size_t count, CupyUtil::py_tensor_map_t tensors);

    /**
     * @brief Get the count object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t get_count(TensorMemory& self);

    /**
     * @brief Get the tensors converted to CuPy arrays. Pybind11 will convert the std::map to a Python dict.
     *
     * @param self
     * @return py_tensor_map_t
     */
    static CupyUtil::py_tensor_map_t get_tensors(TensorMemory& self);

    static void set_tensors(TensorMemory& self, CupyUtil::py_tensor_map_t tensors);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
