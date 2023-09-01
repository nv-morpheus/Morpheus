/*
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
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations********************/
/****** MultiInferenceMessage*******************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * This is a container class that holds a pointer to an instance of the TensorMemory container and the metadata
 * of the data contained within it. Builds on top of the `MultiInferenceMessage` and `MultiTensorMessage` class
 * to add additional data for inferencing.
 */
#pragma GCC visibility push(default)
class MultiInferenceMessage : public DerivedMultiMessage<MultiInferenceMessage, MultiTensorMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiInferenceMessage(const MultiInferenceMessage& other) = default;
    /**
     * @brief Construct a new Multi Inference Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     */
    MultiInferenceMessage(std::shared_ptr<MessageMeta> meta,
                          TensorIndex mess_offset              = 0,
                          TensorIndex mess_count               = -1,
                          std::shared_ptr<TensorMemory> memory = nullptr,
                          TensorIndex offset                   = 0,
                          TensorIndex count                    = -1,
                          std::string id_tensor_name           = "seq_ids");

    /**
     * @brief Returns the input tensor for the given `name`.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    const TensorObject get_input(const std::string& name) const;

    /**
     * @brief Returns the input tensor for the given `name`.
     *
     * @param name
     * @return TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    TensorObject get_input(const std::string& name);

    /**
     * Update the value of ain input tensor. The tensor must already exist, otherwise this will halt on a fatal error.
     */
    void set_input(const std::string& name, const TensorObject& value);
};

/****** MultiInferenceMessageInterfaceProxy****************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceMessageInterfaceProxy : public MultiTensorMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceMessage object, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @return std::shared_ptr<MultiInferenceMessage>
     */
    static std::shared_ptr<MultiInferenceMessage> init(std::shared_ptr<MessageMeta> meta,
                                                       TensorIndex mess_offset,
                                                       TensorIndex mess_count,
                                                       std::shared_ptr<TensorMemory> memory,
                                                       TensorIndex offset,
                                                       TensorIndex count,
                                                       std::string id_tensor_name);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
