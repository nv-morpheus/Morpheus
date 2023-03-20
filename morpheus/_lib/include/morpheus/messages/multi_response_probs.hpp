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

#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for TensorIndex

#include <pybind11/pytypes.h>

#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseProbsMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
 * array. Helps ensure the proper outputs are set and eases debugging
 *
 */
#pragma GCC visibility push(default)
class MultiResponseProbsMessage : public DerivedMultiMessage<MultiResponseProbsMessage, MultiResponseMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiResponseProbsMessage(const MultiResponseProbsMessage& other) = default;

    /**
     * Construct a new Multi Response Probs Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @param probs_tensor_name Name of the tensor that holds output probabilities
     */
    MultiResponseProbsMessage(std::shared_ptr<MessageMeta> meta,
                              TensorIndex mess_offset              = 0,
                              TensorIndex mess_count               = -1,
                              std::shared_ptr<TensorMemory> memory = nullptr,
                              TensorIndex offset                   = 0,
                              TensorIndex count                    = -1,
                              std::string id_tensor_name           = "seq_ids",
                              std::string probs_tensor_name        = "probs");

    /**
     * @brief Returns the `probs` (probabilities) output tensor
     *
     * @return const TensorObject
     */
    const TensorObject get_probs() const;

    /**
     * @brief Update the `probs` output tensor. Will halt on a fatal error if the `probs` output tensor does not exist.
     *
     * @param probs
     */
    void set_probs(const TensorObject& probs);
};

/****** MultiResponseProbsMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseProbsMessageInterfaceProxy : public MultiResponseMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseProbsMessage object, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @param id_tensor_name Name of the tensor that correlates tensor rows to message IDs
     * @param probs_tensor_name Name of the tensor that holds output probabilities
     * @return std::shared_ptr<MultiResponseProbsMessage>
     */
    static std::shared_ptr<MultiResponseProbsMessage> init(std::shared_ptr<MessageMeta> meta,
                                                           TensorIndex mess_offset,
                                                           TensorIndex mess_count,
                                                           std::shared_ptr<TensorMemory> memory,
                                                           TensorIndex offset,
                                                           TensorIndex count,
                                                           std::string id_tensor_name,
                                                           std::string probs_tensor_name);

    /**
     * @brief Return the `probs` (probabilities) output tensor
     *
     * @param self
     * @return pybind11::object
     * @throws pybind11::attribute_error When no tensor named "probs" exists.
     */
    static pybind11::object probs(MultiResponseProbsMessage& self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
