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

#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiInferenceNLPMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiInferenceNLPMessage : public MultiInferenceMessage
{
  public:
    MultiInferenceNLPMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             std::size_t mess_offset,
                             std::size_t mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             std::size_t offset,
                             std::size_t count);

    /**
     * TODO(Documentation)
     */
    const TensorObject get_input_ids() const;

    /**
     * TODO(Documentation)
     */
    void set_input_ids(const TensorObject& input_ids);

    /**
     * TODO(Documentation)
     */
    const TensorObject get_input_mask() const;

    /**
     * TODO(Documentation)
     */
    void set_input_mask(const TensorObject& input_mask);

    /**
     * TODO(Documentation)
     */
    const TensorObject get_seq_ids() const;

    /**
     * TODO(Documentation)
     */
    void set_seq_ids(const TensorObject& seq_ids);
};

/****** MultiInferenceNLPMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceNLPMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceNLPMessage, and return a shared pointer to the result.
     */
    static std::shared_ptr<MultiInferenceNLPMessage> init(std::shared_ptr<MessageMeta> meta,
                                                          cudf::size_type mess_offset,
                                                          cudf::size_type mess_count,
                                                          std::shared_ptr<InferenceMemory> memory,
                                                          cudf::size_type offset,
                                                          cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<morpheus::InferenceMemory> memory(MultiInferenceNLPMessage& self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiInferenceNLPMessage& self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiInferenceNLPMessage& self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object input_ids(MultiInferenceNLPMessage& self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object input_mask(MultiInferenceNLPMessage& self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object seq_ids(MultiInferenceNLPMessage& self);
};
#pragma GCC visibility pop
}  // namespace morpheus
