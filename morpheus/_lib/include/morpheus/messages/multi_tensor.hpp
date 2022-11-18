/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/objects/tensor_object.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>  // for pair
#include <vector>

namespace morpheus {
#pragma GCC visibility push(default)

/****** MultiTensorMessage*******************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * Base class for MultiInferenceMessage & MultiResponseMessage
 * Contains a pointer to an instance of TensorMemory along with an
 * offset & count to those tensors
 *
 * mess_offset & mess_count refer to the range of records in meta.
 * offset & count refer to the range of records in TensorMemory
 *
 * While TensorMemory can contain multiple tensors, it is a requirement that
 * they are all of the same length and that element N in each tensor refers
 * to the same record
 *
 */
class MultiTensorMessage : public DerivedMultiMessage<MultiTensorMessage, MultiMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiTensorMessage(const MultiTensorMessage &other) = default;

    /**
     * Construct a new Multi Tensor Message object.
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in tensor memory instance
     * @param count Message count in tensor memory instance
     */
    MultiTensorMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                       std::size_t mess_offset,
                       std::size_t mess_count,
                       std::shared_ptr<morpheus::TensorMemory> memory,
                       std::size_t offset,
                       std::size_t count);

    std::shared_ptr<morpheus::TensorMemory> memory;
    std::size_t offset{0};
    std::size_t count{0};

    /**
     * @brief Returns a tensor with the given name. Will halt on a fatal error if the tensor does not exist
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_tensor(const std::string &name) const;

    /**
     * @brief Returns a tensor with the given name. Will halt on a fatal error if the tensor does not exist
     *
     * @param name
     * @return TensorObject
     */
    TensorObject get_tensor(const std::string &name);

    /**
     * @brief Update the value of a given tensor. The tensor must already exist, otherwise this will halt on a fatal
     * error
     *
     * @param name
     * @param value
     */
    void set_tensor(const std::string &name, const TensorObject &value);

  protected:
    void get_slice_impl(std::shared_ptr<MultiMessage> new_message, std::size_t start, std::size_t stop) const override;

    void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                          const std::vector<std::pair<std::size_t, std::size_t>> &ranges,
                          size_t num_selected_rows) const override;

    std::shared_ptr<morpheus::TensorMemory> copy_input_ranges(
        const std::vector<std::pair<std::size_t, std::size_t>> &ranges, std::size_t num_selected_rows) const;

    TensorObject get_tensor_impl(const std::string &name) const;
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
