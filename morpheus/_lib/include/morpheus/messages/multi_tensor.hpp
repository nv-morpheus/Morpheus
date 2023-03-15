/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/types.hpp"  // for TensorIndex, RangeType

#include <pybind11/pytypes.h>  // for object

#include <memory>
#include <optional>
#include <string>
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
    MultiTensorMessage(const MultiTensorMessage& other) = default;

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
    MultiTensorMessage(std::shared_ptr<MessageMeta> meta,
                       TensorIndex mess_offset = 0,
                       TensorIndex mess_count = -1,
                       std::shared_ptr<TensorMemory> memory = nullptr,
                       TensorIndex offset = 0,
                       TensorIndex count = -1);

    std::shared_ptr<morpheus::TensorMemory> memory;
    TensorIndex offset{0};
    TensorIndex count{0};

    /**
     * @brief Returns a tensor with the given name.
     *
     * @param name
     * @return const TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    const TensorObject get_tensor(const std::string& name) const;

    /**
     * @brief Returns a tensor with the given name.
     *
     * @param name
     * @return TensorObject
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    TensorObject get_tensor(const std::string& name);

    /**
     * @brief Update the value of a given tensor. The tensor must already exist, otherwise a runtime_error is thrown.
     * error
     *
     * @param name
     * @param value
     * @throws std::runtime_error If no tensor matching `name` exists
     */
    void set_tensor(const std::string& name, const TensorObject& value);

  protected:
    void get_slice_impl(std::shared_ptr<MultiMessage> new_message, TensorIndex start, TensorIndex stop) const override;

    void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                          const std::vector<RangeType>& ranges,
                          TensorIndex num_selected_rows) const override;

    std::shared_ptr<morpheus::TensorMemory> copy_input_ranges(const std::vector<RangeType>& ranges,
                                                              TensorIndex num_selected_rows) const;

    TensorObject get_tensor_impl(const std::string& name) const;
};

/****** MultiTensorMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiTensorMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiTensorMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Shared pointer of a tensor memory
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @return std::shared_ptr<MultiTensorMessage>
     */
    static std::shared_ptr<MultiTensorMessage> init(std::shared_ptr<MessageMeta> meta,
                                                    TensorIndex mess_offset,
                                                    TensorIndex mess_count,
                                                    std::shared_ptr<TensorMemory> memory,
                                                    TensorIndex offset,
                                                    TensorIndex count);

    /**
     * @brief Returns a shared pointer of a tensor memory object
     *
     * @return std::shared_ptr<TensorMemory>
     */
    static std::shared_ptr<TensorMemory> memory(MultiTensorMessage& self);

    /**
     * @brief Message offset in tensor memory object
     *
     * @param self
     * @return TensorIndex
     */
    static TensorIndex offset(MultiTensorMessage& self);

    /**
     * @brief Messages count in tensor memory object
     *
     * @param self
     * @return TensorIndex
     */
    static TensorIndex count(MultiTensorMessage& self);

    /**
     * @brief Returns the tensor tensor for a given name
     *
     * @param self
     * @param name : Tensor name
     * @return pybind11::object
     * @throws pybind11::key_error When no matching tensor exists.
     */
    static pybind11::object get_tensor(MultiTensorMessage& self, const std::string& name);

    /**
     * @brief Same as `get_tensor` but used when the method is being bound to a python property
     *
     * @param self
     * @param name
     * @return pybind11::object
     * @throws pybind11::attribute_error When no matching tensor exists.
     */
    static pybind11::object get_tensor_property(MultiTensorMessage& self, const std::string name);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
