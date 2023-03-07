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

#include "morpheus/messages/memory/inference_memory.hpp"  // for InferenceMemory
#include "morpheus/messages/meta.hpp"                     // for MessageMeta
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>  // for object

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiInferenceNLPMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
 * proper inputs are set and eases debugging.
 *
 */
class MultiInferenceNLPMessage : public MultiInferenceMessage
{
  public:
    /**
     * @brief Construct a new Multi Inference NLP Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     */
    MultiInferenceNLPMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             TensorIndex mess_offset,
                             TensorIndex mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             TensorIndex offset,
                             TensorIndex count);

    /**
     * @brief Returns the 'input_ids' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor named "input_ids" exists
     */
    const TensorObject get_input_ids() const;

    /**
     * @brief Sets a tensor named 'input_ids'.
     *
     * @param input_ids
     */
    void set_input_ids(const TensorObject& input_ids);

    /**
     * @brief Returns the 'input_mask' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor named "input_mask" exists
     */
    const TensorObject get_input_mask() const;

    /**
     * @brief Sets a tensor named 'input_mask'.
     *
     * @param input_mask
     */
    void set_input_mask(const TensorObject& input_mask);

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor named "seq_ids" exists
     */
    const TensorObject get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'seq_ids'.
     *
     * @param seq_ids
     */
    void set_seq_ids(const TensorObject& seq_ids);
};

/****** MultiInferenceNLPMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceNLPMessageInterfaceProxy : public MultiInferenceMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceNLPMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     * @return std::shared_ptr<MultiInferenceNLPMessage>
     */
    static std::shared_ptr<MultiInferenceNLPMessage> init(std::shared_ptr<MessageMeta> meta,
                                                          TensorIndex mess_offset,
                                                          TensorIndex mess_count,
                                                          std::shared_ptr<InferenceMemory> memory,
                                                          TensorIndex offset,
                                                          TensorIndex count);

    /**
     * @brief Get  'input_ids' tensor as a python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "input_ids" exists.
     */
    static pybind11::object input_ids(MultiInferenceNLPMessage& self);

    /**
     * @brief Get 'input_mask' tensor as a python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "input_mask" exists.
     */
    static pybind11::object input_mask(MultiInferenceNLPMessage& self);

    /**
     * @brief Get 'seq_ids' tensor as a python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "seq_ids" exists.
     */
    static pybind11::object seq_ids(MultiInferenceNLPMessage& self);
};
#pragma GCC visibility pop
}  // namespace morpheus
