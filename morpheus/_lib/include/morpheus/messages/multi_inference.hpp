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

#include "morpheus/messages/memory/inference_memory.hpp"
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
/****** Component public implementations********************/
/****** MultiInferenceMessage*******************************/
/**
 * This is a container class that holds a pointer to an instance of the TensorMemory container and the metadata 
 * of the data contained within it. Builds on top of the `MultiInferenceMessage` and `MultiTensorMessage` class 
 * to add additional data for inferencing.
 * 
 * 
 * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and 
 * C++ representations of the table
 * @param mess_offset Offset into the metadata batch
 * @param mess_count Messages count
 * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
 * @param offset Message offset in inference memory instance
 * @param count Message count in inference memory instance
 */
#pragma GCC visibility push(default)
class MultiInferenceMessage : public DerivedMultiMessage<MultiInferenceMessage, MultiTensorMessage>
{
  public:
    MultiInferenceMessage(const MultiInferenceMessage &other) = default;
    MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                          std::size_t mess_offset,
                          std::size_t mess_count,
                          std::shared_ptr<morpheus::InferenceMemory> memory,
                          std::size_t offset,
                          std::size_t count);

    /**
     * @brief Return the input tensor for the given `name`. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_input(const std::string &name) const;

    /**
     * @brief Return the input tensor for the given `name`. Will halt on a fatal error if the tensor does not exist.
     *
     * @param name
     * @return TensorObject
     */
    TensorObject get_input(const std::string &name);

    /**
     * Update the value of ain input tensor. The tensor must already exist, otherwise this will halt on a fatal error.
     */
    void set_input(const std::string &name, const TensorObject &value);
};

/****** MultiInferenceMessageInterfaceProxy****************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceMessage object, and return a shared pointer to the result.
     */
    static std::shared_ptr<MultiInferenceMessage> init(std::shared_ptr<MessageMeta> meta,
                                                       cudf::size_type mess_offset,
                                                       cudf::size_type mess_count,
                                                       std::shared_ptr<InferenceMemory> memory,
                                                       cudf::size_type offset,
                                                       cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<morpheus::InferenceMemory> memory(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_input(MultiInferenceMessage &self, const std::string &name);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MultiInferenceMessage> get_slice(MultiInferenceMessage &self,
                                                            std::size_t start,
                                                            std::size_t stop);
};
#pragma GCC visibility pop
}  // namespace morpheus
