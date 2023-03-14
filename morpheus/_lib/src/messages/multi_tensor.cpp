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

#include "morpheus/messages/multi_tensor.hpp"

#include "morpheus/types.hpp"                // for TensorIndex, TensorMap
#include "morpheus/utilities/cupy_util.hpp"  // for CupyUtil::tensor_to_cupy
#include "morpheus/utilities/string_util.hpp"

#include <cudf/types.hpp>  // for cudf::size_type>
#include <glog/logging.h>
#include <pybind11/pytypes.h>  // for key_error

#include <cstdint>    // for int32_t
#include <stdexcept>  // for runtime_error

namespace morpheus {
/****** Component public implementations *******************/
/****** <MultiTensorMessage>****************************************/
MultiTensorMessage::MultiTensorMessage(std::shared_ptr<MessageMeta> meta,
                                       size_t mess_offset,
                                       std::optional<size_t> mess_count,
                                       std::shared_ptr<TensorMemory> memory,
                                       size_t offset,
                                       std::optional<size_t> count) :
  DerivedMultiMessage(meta, mess_offset, mess_count),
  memory(std::move(memory)),
  offset(offset)
{
    if (!this->memory)
    {
        throw std::invalid_argument("Must define `memory` when creating MultiTensorMessage");
    }

    // Default to using the count from the meta if it is unset
    this->count = count.value_or(this->memory->count);

    if (this->offset < 0 || this->offset >= this->memory->count)
    {
        throw std::invalid_argument("Invalid offset value");
    }
    if (this->count <= 0 || (this->offset + this->count > this->memory->count))
    {
        throw std::invalid_argument("Invalid count value");
    }
}

const TensorObject MultiTensorMessage::get_tensor(const std::string& name) const
{
    return get_tensor_impl(name);
}

TensorObject MultiTensorMessage::get_tensor(const std::string& name)
{
    return get_tensor_impl(name);
}

TensorObject MultiTensorMessage::get_tensor_impl(const std::string& name) const
{
    auto& tensor = this->memory->get_tensor(name);

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return tensor;
    }

    return tensor.slice({static_cast<cudf::size_type>(this->offset), 0},
                        {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

void MultiTensorMessage::set_tensor(const std::string& name, const TensorObject& value)
{
    // Get the input slice first
    auto slice = this->get_tensor(name);

    // Set the value to use assignment
    slice = value;
}

void MultiTensorMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                        std::size_t start,
                                        std::size_t stop) const
{
    DCHECK(std::dynamic_pointer_cast<MultiTensorMessage>(new_message) != nullptr);
    auto sliced_message = std::static_pointer_cast<MultiTensorMessage>(new_message);

    // Start must be between [0, mess_count)
    if (start < 0 || start >= this->count)
    {
        throw std::out_of_range("Invalid memory `start` argument");
    }

    // Stop must be between (start, mess_count]
    if (stop <= start or stop > this->count)
    {
        throw std::out_of_range("Invalid memory `stop` argument");
    }

    sliced_message->offset = start;
    sliced_message->count  = stop - start;

    if (this->count != this->mess_count)
    {
        // If we have more tensor rows than message rows, we need to use the seq_ids to figure out the slicing. This
        // will be slow and should be avoided at all costs
        if (!this->memory->has_tensor("seq_ids"))
        {
            throw std::runtime_error(
                "The tensor memory object is missing the required ID tensor 'seq_ids' this tensor is required to make "
                "slices of MultiTensorMessages");
        }

        auto id_tensor = this->get_tensor("seq_ids");

        // Determine the new start and stop before passing onto the base
        start = id_tensor.read_element<int32_t>({(TensorIndex)start, 0});
        stop  = id_tensor.read_element<int32_t>({(TensorIndex)stop - 1, 0}) + 1;
    }

    // Pass onto the base
    DerivedMultiMessage::get_slice_impl(new_message, start, stop);
}

void MultiTensorMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                          const std::vector<std::pair<size_t, size_t>>& ranges,
                                          size_t num_selected_rows) const
{
    DCHECK(std::dynamic_pointer_cast<MultiTensorMessage>(new_message) != nullptr);
    auto copied_message = std::static_pointer_cast<MultiTensorMessage>(new_message);
    DerivedMultiMessage::copy_ranges_impl(copied_message, ranges, num_selected_rows);

    copied_message->offset = 0;
    copied_message->count  = num_selected_rows;
    copied_message->memory = copy_input_ranges(ranges, num_selected_rows);
}

std::shared_ptr<TensorMemory> MultiTensorMessage::copy_input_ranges(
    const std::vector<std::pair<size_t, size_t>>& ranges, size_t num_selected_rows) const
{
    auto offset_ranges = apply_offset_to_ranges(offset, ranges);
    auto tensors       = memory->copy_tensor_ranges(offset_ranges, num_selected_rows);
    return std::make_shared<TensorMemory>(num_selected_rows, std::move(tensors));
}

/****** MultiTensorMessageInterfaceProxy *************************/
std::shared_ptr<MultiTensorMessage> MultiTensorMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                                           size_t mess_offset,
                                                                           std::optional<size_t> mess_count,
                                                                           std::shared_ptr<TensorMemory> memory,
                                                                           size_t offset,
                                                                           std::optional<size_t> count)
{
    return std::make_shared<MultiTensorMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::TensorMemory> MultiTensorMessageInterfaceProxy::memory(MultiTensorMessage& self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::TensorMemory>(self.memory) != nullptr);

    return std::static_pointer_cast<morpheus::TensorMemory>(self.memory);
}

std::size_t MultiTensorMessageInterfaceProxy::offset(MultiTensorMessage& self)
{
    return self.offset;
}

std::size_t MultiTensorMessageInterfaceProxy::count(MultiTensorMessage& self)
{
    return self.count;
}

pybind11::object MultiTensorMessageInterfaceProxy::get_tensor(MultiTensorMessage& self, const std::string& name)
{
    try
    {
        auto tensor = self.get_tensor(name);
        return CupyUtil::tensor_to_cupy(tensor);
    } catch (const std::runtime_error& e)
    {
        throw pybind11::key_error{e.what()};
    }
}

pybind11::object MultiTensorMessageInterfaceProxy::get_tensor_property(MultiTensorMessage& self, const std::string name)
{
    try
    {
        return get_tensor(self, std::move(name));
    } catch (const pybind11::key_error& e)
    {
        throw pybind11::attribute_error{e.what()};
    }
}

}  // namespace morpheus
