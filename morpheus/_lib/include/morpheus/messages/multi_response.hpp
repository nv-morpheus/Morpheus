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
#include <morpheus/objects/table_info.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/objects/tensor_object.hpp>
#include <morpheus/utilities/table_util.hpp>

#include <cudf/types.hpp>

#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiResponseMessage : public DerivedMultiMessage<MultiResponseMessage, MultiMessage>
{
  public:
    MultiResponseMessage(std::shared_ptr<MessageMeta> meta,
                         std::size_t mess_offset,
                         std::size_t mess_count,
                         std::shared_ptr<ResponseMemory> memory,
                         std::size_t offset,
                         std::size_t count);

    std::shared_ptr<ResponseMemory> memory;
    std::size_t offset{0};
    std::size_t count{0};

    /**
     * TODO(Documentation)
     */
    TensorObject get_output(const std::string &name);

    /**
     * TODO(Documentation)
     */
    const TensorObject get_output(const std::string &name) const;

    /**
     * TODO(Documentation)
     */
    const void set_output(const std::string &name, const TensorObject &value);

    // /**
    //  * @brief Creates a copy of the current message calculating new `mess_offset` and `mess_count` values based on
    //  the
    //  * given `start` & `stop` values. This method is reletively light-weight as it does not copy the underlying
    //  `meta`
    //  * or `memory` objects. The actual slicing of each is applied later when `get_meta` and `get_output` is called.
    //  *
    //  * @param start
    //  * @param stop
    //  * @return std::shared_ptr<MultiResponseMessage>
    //  */
    // std::shared_ptr<MultiResponseMessage> get_slice(std::size_t start, std::size_t stop) const;

    /**
     * @brief Creates a deep copy of the current message along with a copy of the underlying `meta` and `memory`
     * selecting the rows of both. Defined by pairs of start, stop rows expressed in the `ranges` argument.
     *
     * This allows for copying several non-contiguous rows from the underlying dataframe and tensors into a new objects,
     * however this comes at a much higher cost compared to the `get_slice` method.
     *
     * @param ranges
     * @param num_selected_rows
     * @return std::shared_ptr<MultiResponseMessage>
     */
    std::shared_ptr<MultiResponseMessage> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                      size_t num_selected_rows) const;

  protected:
    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MultiMessage> get_slice_impl(std::size_t start, std::size_t stop) const override;

    std::shared_ptr<MultiMessage> internal_copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                       size_t num_selected_rows) const override;

    std::shared_ptr<ResponseMemory> copy_output_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                       size_t num_selected_rows) const;
};

/****** MultiResponseMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseMessage, and return a shared pointer to the result.
     */
    static std::shared_ptr<MultiResponseMessage> init(std::shared_ptr<MessageMeta> meta,
                                                      cudf::size_type mess_offset,
                                                      cudf::size_type mess_count,
                                                      std::shared_ptr<ResponseMemory> memory,
                                                      cudf::size_type offset,
                                                      cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<ResponseMemory> memory(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiResponseMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_output(MultiResponseMessage &self, const std::string &name);
};
#pragma GCC visibility pop
}  // namespace morpheus
