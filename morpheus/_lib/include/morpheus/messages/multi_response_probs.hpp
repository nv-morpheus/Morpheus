/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/messages/memory/response_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/messages/multi_response.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/objects/tensor_object.hpp>

#include <pybind11/pytypes.h>
#include <cudf/types.hpp>

#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseProbsMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiResponseProbsMessage : public MultiResponseMessage
{
  public:
    MultiResponseProbsMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                              size_t mess_offset,
                              size_t mess_count,
                              std::shared_ptr<morpheus::ResponseMemory> memory,
                              size_t offset,
                              size_t count);

    /**
     * TODO(Documentation)
     */
    const TensorObject get_probs() const;

    /**
     * TODO(Documentation)
     */
    void set_probs(const TensorObject &probs);

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MultiResponseProbsMessage> get_slice(size_t start, size_t stop) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiResponseProbsMessage>(this->internal_get_slice(start, stop));
    }
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
                                                           std::shared_ptr<ResponseMemory> memory,
                                                           cudf::size_type offset,
                                                           cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<morpheus::ResponseMemory> memory(MultiResponseProbsMessage &self);

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
