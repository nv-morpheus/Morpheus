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
#include "morpheus/messages/meta.hpp"  // for MessageMeta
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <pybind11/pytypes.h>  // for object

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiInferenceFILMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
 * proper inputs are set and eases debugging.
 *
 */
#pragma GCC visibility push(default)
class MultiInferenceFILMessage : public DerivedMultiMessage<MultiInferenceFILMessage, MultiInferenceMessage>
{
  public:
    /**
     * @brief Construct a new Multi Inference FIL Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     */
    MultiInferenceFILMessage(std::shared_ptr<MessageMeta> meta,
                             TensorIndex mess_offset              = 0,
                             TensorIndex mess_count               = -1,
                             std::shared_ptr<TensorMemory> memory = nullptr,
                             TensorIndex offset                   = 0,
                             TensorIndex count                    = -1,
                             std::string id_tensor_name           = "seq_ids");

    /**
     * @brief Returns the 'input__0' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor named "input__0" exists
     */
    const TensorObject get_input__0() const;

    /**
     * @brief Sets a tensor named 'input__0'
     *
     * @param input__0
     */
    void set_input__0(const TensorObject& input__0);

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor named "seq_ids" exists
     */
    const TensorObject get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'seq_ids'
     *
     * @param seq_ids
     */
    void set_seq_ids(const TensorObject& seq_ids);
};

/****** MultiInferenceFILMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceFILMessageInterfaceProxy : public MultiInferenceMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceFILMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @return std::shared_ptr<MultiInferenceFILMessage>
     */
    static std::shared_ptr<MultiInferenceFILMessage> init(std::shared_ptr<MessageMeta> meta,
                                                          TensorIndex mess_offset,
                                                          TensorIndex mess_count,
                                                          std::shared_ptr<TensorMemory> memory,
                                                          TensorIndex offset,
                                                          TensorIndex count,
                                                          std::string id_tensor_name);

    /**
     * @brief Get  'input__0' tensor as a python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "input__0" exists.
     */
    static pybind11::object input__0(MultiInferenceFILMessage& self);

    /**
     * @brief Get 'seq_ids' tensor as a python object
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "seq_ids" exists.
     */
    static pybind11::object seq_ids(MultiInferenceFILMessage& self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
