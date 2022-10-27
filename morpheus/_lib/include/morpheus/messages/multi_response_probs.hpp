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

#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>  // for size_t
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseProbsMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiResponseProbsMessage : public DerivedMultiMessage<MultiResponseProbsMessage, MultiResponseMessage>
{
  public:
    MultiResponseProbsMessage(const MultiResponseProbsMessage &other) = default;
    MultiResponseProbsMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                              size_t mess_offset,
                              size_t mess_count,
                              std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                              size_t offset,
                              size_t count);

    /**
     * @brief Return the `probs` (probabilities) output tensor
     *
     * @return const TensorObject
     */
    const TensorObject get_probs() const;

    /**
     * @brief Update the `probs` output tensor. Will halt on a fatal error if the `probs` output tensor does not exist.
     *
     * @param probs
     */
    void set_probs(const TensorObject &probs);
};

/****** MultiResponseProbsMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseProbsMessageInterfaceProxy
{
    static std::shared_ptr<MultiResponseProbsMessage> init(std::shared_ptr<MessageMeta> meta,
                                                           cudf::size_type mess_offset,
                                                           cudf::size_type mess_count,
                                                           std::shared_ptr<ResponseMemoryProbs> memory,
                                                           cudf::size_type offset,
                                                           cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<morpheus::ResponseMemoryProbs> memory(MultiResponseProbsMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiResponseProbsMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiResponseProbsMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object probs(MultiResponseProbsMessage &self);
};
#pragma GCC visibility pop
}  // namespace morpheus
