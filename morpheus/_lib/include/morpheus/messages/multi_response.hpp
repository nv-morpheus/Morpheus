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
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>  // for pair
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiResponseMessage : public DerivedMultiMessage<MultiResponseMessage, MultiTensorMessage>
{
  public:
    MultiResponseMessage(const MultiResponseMessage &other) = default;
    MultiResponseMessage(std::shared_ptr<MessageMeta> meta,
                         std::size_t mess_offset,
                         std::size_t mess_count,
                         std::shared_ptr<ResponseMemory> memory,
                         std::size_t offset,
                         std::size_t count) :
      DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count){};

    /**
     * @brief Returns the output tensor with the given name. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_output(const std::string &name) const
    {
        return get_tensor(name);
    };

    /**
     * @brief Returns the output tensor with the given name. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return TensorObject
     */
    TensorObject get_output(const std::string &name)
    {
        return get_tensor(name);
    };

    /**
     * @brief Update the value of a given output tensor. The tensor must already exist, otherwise this will halt on a
     * fatal error.
     *
     * @param name
     * @param value
     */
    void set_output(const std::string &name, const TensorObject &value)
    {
        set_tensor(name, value);
    };
};

/****** MultiResponseMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseMessage, and return a shared pointer to the result.
     */
    static std::shared_ptr<MultiResponseMessage> init(std::shared_ptr<MessageMeta> meta,
                                                      cudf::size_type mess_offset,
                                                      cudf::size_type mess_count,
                                                      std::shared_ptr<ResponseMemory> memory,
                                                      cudf::size_type offset,
                                                      cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<ResponseMemory> memory(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_output(MultiResponseMessage &self, const std::string &name);
};
#pragma GCC visibility pop
}  // namespace morpheus
