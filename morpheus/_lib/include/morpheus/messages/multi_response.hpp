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
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * This class is used to get or set the inference output from message containers derived
 * from ResponseMemory.
 *
 */
#pragma GCC visibility push(default)
class MultiResponseMessage : public DerivedMultiMessage<MultiResponseMessage, MultiTensorMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiResponseMessage(const MultiResponseMessage& other) = default;

    /**
     * @brief Construct a new Multi Response Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     */
    MultiResponseMessage(std::shared_ptr<MessageMeta> meta,
                         std::size_t mess_offset,
                         std::size_t mess_count,
                         std::shared_ptr<ResponseMemory> memory,
                         std::size_t offset,
                         std::size_t count);

    /**
     * @brief Returns the output tensor with the given name. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_output(const std::string& name) const;

    /**
     * @brief Returns the output tensor with the given name. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return TensorObject
     */
    TensorObject get_output(const std::string& name);

    /**
     * @brief Update the value of a given output tensor. The tensor must already exist, otherwise this will halt on a
     * fatal error.
     *
     * @param name
     * @param value
     */
    void set_output(const std::string& name, const TensorObject& value);
};

/****** MultiResponseMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @return std::shared_ptr<MultiResponseMessage>
     */
    static std::shared_ptr<MultiResponseMessage> init(std::shared_ptr<MessageMeta> meta,
                                                      cudf::size_type mess_offset,
                                                      cudf::size_type mess_count,
                                                      std::shared_ptr<ResponseMemory> memory,
                                                      cudf::size_type offset,
                                                      cudf::size_type count);

    /**
     * @brief GReturns a shared pointer of a response memory probs object
     *
     * @return std::shared_ptr<ResponseMemory>
     */
    static std::shared_ptr<ResponseMemory> memory(MultiResponseMessage& self);

    /**
     * @brief Message offset in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t offset(MultiResponseMessage& self);

    /**
     * @brief Messages count in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(MultiResponseMessage& self);

    /**
     * @brief Returns the output tensor for a given name
     *
     * @param self
     * @param name : Tensor name
     * @return pybind11::object
     */
    static pybind11::object get_output(MultiResponseMessage& self, const std::string& name);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
