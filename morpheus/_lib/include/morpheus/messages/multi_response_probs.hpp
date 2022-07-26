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

#include <morpheus/messages/memory/response_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/messages/multi_response.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/objects/tensor_object.hpp>

#include <pybind11/pytypes.h>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

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

    // /**
    //  * @brief Creates a copy of the current message calculating new `mess_offset` and `mess_count` values based on
    //  the
    //  * given `start` & `stop` values. This method is reletively light-weight as it does not copy the underlying
    //  `meta`
    //  * or `memory` objects. The actual slicing of each is applied later when `get_meta` and `get_output` is called.
    //  *
    //  * @param start
    //  * @param stop
    //  * @return std::shared_ptr<MultiResponseProbsMessage>
    //  */
    // std::shared_ptr<MultiResponseProbsMessage> get_slice(size_t start, size_t stop) const
    // {
    //     // This can only cast down
    //     return std::static_pointer_cast<MultiResponseProbsMessage>(this->internal_get_slice(start, stop));
    // }

    /**
     * @brief Creates a deep copy of the current message along with a copy of the underlying `meta` and `memory`
     * selecting the rows of both. Defined by pairs of start, stop rows expressed in the `ranges` argument.
     *
     * This allows for copying several non-contiguous rows from the underlying dataframe and tensors into a new objects,
     * however this comes at a much higher cost compared to the `get_slice` method.
     *
     * @param ranges
     * @param num_selected_rows
     * @return std::shared_ptr<MultiResponseProbsMessage>
     */
    std::shared_ptr<MultiResponseProbsMessage> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                           size_t num_selected_rows) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiResponseProbsMessage>(
            this->internal_copy_ranges(ranges, num_selected_rows));
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
