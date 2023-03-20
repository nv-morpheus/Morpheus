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

#include "morpheus/objects/dtype.hpp"
#include "morpheus/types.hpp"                // for TensorIndex, TensorMap
#include "morpheus/utilities/cupy_util.hpp"  // for CupyUtil::tensor_to_cupy
#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>        // IWYU pragma: keep
#include <mrc/utils/macros.hpp>  // for MRC_PTR_CAST
#include <pybind11/pytypes.h>    // for key_error

#include <cstdint>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <utility>    // for move

namespace morpheus {

// MatX works best with C-Style arrays so ignore this warning
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
TensorIndex read_idx_from_tensor(const TensorObject& tensor, const TensorIndex (&idx)[2])
{
    switch (tensor.dtype().type_id())
    {
    case TypeId::INT8:
        return tensor.read_element<int8_t>(idx);
    case TypeId::INT16:
        return tensor.read_element<int16_t>(idx);
    case TypeId::INT32:
        return tensor.read_element<int32_t>(idx);
    case TypeId::INT64:
        return tensor.read_element<int64_t>(idx);
    case TypeId::UINT8:
        return tensor.read_element<uint8_t>(idx);
    case TypeId::UINT16:
        return tensor.read_element<uint16_t>(idx);
    case TypeId::UINT32:
        return tensor.read_element<uint32_t>(idx);
    case TypeId::UINT64:
        return tensor.read_element<uint64_t>(idx);
    default:
        CHECK(false) << "Unsupported index type" << tensor.dtype().type_str();
        return -1;
    }
}

/****** Component public implementations *******************/
/****** <MultiTensorMessage>****************************************/
MultiTensorMessage::MultiTensorMessage(std::shared_ptr<MessageMeta> meta,
                                       TensorIndex mess_offset,
                                       TensorIndex mess_count,
                                       std::shared_ptr<TensorMemory> memory,
                                       TensorIndex offset,
                                       TensorIndex count,
                                       std::string id_tensor_name) :
  DerivedMultiMessage(meta, mess_offset, mess_count),
  memory(std::move(memory)),
  offset(offset),
  id_tensor_name(std::move(id_tensor_name))
{
    if (!this->memory)
    {
        throw std::invalid_argument("Must define `memory` when creating MultiTensorMessage");
    }

    // Default to using the count from the meta if it is unset
    if (count == -1)
    {
        count = this->memory->count - offset;
    }

    this->count = count;

    if (this->offset < 0 || this->offset >= this->memory->count)
    {
        throw std::invalid_argument("Invalid offset value");
    }
    if (this->count <= 0 || (this->offset + this->count > this->memory->count))
    {
        throw std::invalid_argument("Invalid count value");
    }
    if (this->count < this->mess_count)
    {
        throw std::invalid_argument("Invalid count value. Must have a count greater than or equal to mess_count");
    }

    // Finally, perform a consistency check on the seq_ids
    if (this->memory->has_tensor(this->id_tensor_name))
    {
        auto id_tensor = this->memory->get_tensor(this->id_tensor_name);

        TensorIndex first_element = read_idx_from_tensor(id_tensor, {this->offset, 0});
        TensorIndex last_element  = read_idx_from_tensor(id_tensor, {this->offset + this->count - 1, 0});

        if (first_element != this->mess_offset)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Inconsistent ID column. First element in '"
                                                         << this->id_tensor_name << "' tensor, [" << first_element
                                                         << "], must match mess_offset, [" << this->mess_offset
                                                         << "]"));
        }

        if (last_element != this->mess_offset + this->mess_count - 1)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Inconsistent ID column. Last element in '"
                                                         << this->id_tensor_name << "' tensor, [" << last_element
                                                         << "], must not extend beyond last message, ["
                                                         << (this->mess_offset + this->mess_count - 1) << "]"));
        }
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

    return tensor.slice({this->offset, 0}, {this->offset + this->count, -1});
}

void MultiTensorMessage::set_tensor(const std::string& name, const TensorObject& value)
{
    // Get the input slice first
    auto slice = this->get_tensor(name);

    // Set the value to use assignment
    slice = value;
}

const TensorObject MultiTensorMessage::get_id_tensor() const
{
    try
    {
        return this->get_tensor(this->id_tensor_name);
    } catch (std::runtime_error)
    {
        // Throw a better error here if we are missing the ID tensor
        throw pybind11::key_error{MORPHEUS_CONCAT_STR("Cannot get ID tensor. Tensor with name '"
                                                      << this->id_tensor_name
                                                      << "' does not exist in the memory object")};
    }
}

void MultiTensorMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                        TensorIndex start,
                                        TensorIndex stop) const
{
    auto sliced_message = MRC_PTR_CAST(MultiTensorMessage, new_message);

    // Start must be between [0, mess_count)
    if (start < 0 || start >= this->count)
    {
        throw std::out_of_range("Invalid memory `start` argument");
    }

    // Stop must be between (start, mess_count]
    if (stop <= start || stop > this->count)
    {
        throw std::out_of_range("Invalid memory `stop` argument");
    }

    sliced_message->memory         = this->memory;
    sliced_message->offset         = this->offset + start;
    sliced_message->count          = stop - start;
    sliced_message->id_tensor_name = this->id_tensor_name;

    if (this->count != this->mess_count)
    {
        // If we have more tensor rows than message rows, we need to use the seq_ids to figure out the slicing. This
        // will be slow and should be avoided at all costs
        if (!this->memory->has_tensor(this->id_tensor_name))
        {
            throw std::runtime_error(
                "The tensor memory object is missing the required ID tensor 'seq_ids' this tensor is required to make "
                "slices of MultiTensorMessages");
        }

        auto id_tensor = this->get_id_tensor();

        // Determine the new start and stop before passing onto the base
        start = read_idx_from_tensor(id_tensor, {start, 0}) - this->mess_offset;
        stop  = read_idx_from_tensor(id_tensor, {stop - 1, 0}) + 1 - this->mess_offset;
    }

    // Pass onto the base
    DerivedMultiMessage::get_slice_impl(new_message, start, stop);
}

void MultiTensorMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                          const std::vector<RangeType>& ranges,
                                          TensorIndex num_selected_rows) const
{
    auto copied_message = MRC_PTR_CAST(MultiTensorMessage, new_message);
    DerivedMultiMessage::copy_ranges_impl(copied_message, ranges, num_selected_rows);

    copied_message->offset = 0;
    copied_message->count  = num_selected_rows;
    copied_message->memory = copy_input_ranges(ranges, num_selected_rows);
}

std::shared_ptr<TensorMemory> MultiTensorMessage::copy_input_ranges(const std::vector<RangeType>& ranges,
                                                                    TensorIndex num_selected_rows) const
{
    auto offset_ranges = apply_offset_to_ranges(offset, ranges);
    auto tensors       = memory->copy_tensor_ranges(offset_ranges, num_selected_rows);
    return std::make_shared<TensorMemory>(num_selected_rows, std::move(tensors));
}

/****** MultiTensorMessageInterfaceProxy *************************/
std::shared_ptr<MultiTensorMessage> MultiTensorMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                                           TensorIndex mess_offset,
                                                                           TensorIndex mess_count,
                                                                           std::shared_ptr<TensorMemory> memory,
                                                                           TensorIndex offset,
                                                                           TensorIndex count,
                                                                           std::string id_tensor_name)
{
    return std::make_shared<MultiTensorMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count, std::move(id_tensor_name));
}

std::shared_ptr<morpheus::TensorMemory> MultiTensorMessageInterfaceProxy::memory(MultiTensorMessage& self)
{
    return MRC_PTR_CAST(morpheus::TensorMemory, self.memory);
}

TensorIndex MultiTensorMessageInterfaceProxy::offset(MultiTensorMessage& self)
{
    return self.offset;
}

TensorIndex MultiTensorMessageInterfaceProxy::count(MultiTensorMessage& self)
{
    return self.count;
}

std::string MultiTensorMessageInterfaceProxy::id_tensor_name_getter(MultiTensorMessage& self)
{
    return self.id_tensor_name;
}

void MultiTensorMessageInterfaceProxy::id_tensor_name_setter(MultiTensorMessage& self, std::string id_tensor_name)
{
    self.id_tensor_name = id_tensor_name;
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

pybind11::object MultiTensorMessageInterfaceProxy::get_id_tensor(MultiTensorMessage& self)
{
    return CupyUtil::tensor_to_cupy(self.get_id_tensor());
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
