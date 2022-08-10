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

#include "morpheus/messages/multi_response.hpp"

#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/
MultiResponseMessage::MultiResponseMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                           std::size_t mess_offset,
                                           std::size_t mess_count,
                                           std::shared_ptr<ResponseMemory> memory,
                                           std::size_t offset,
                                           std::size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count),
  memory(std::move(memory)),
  offset(offset),
  count(count)
{}

TensorObject MultiResponseMessage::get_output(const std::string &name)
{
    CHECK(this->memory->has_output(name)) << "Could not find output: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->tensors[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->tensors[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const TensorObject MultiResponseMessage::get_output(const std::string &name) const
{
    CHECK(this->memory->has_output(name)) << "Could not find output: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->tensors[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->tensors[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const void MultiResponseMessage::set_output(const std::string &name, const TensorObject &value)
{
    // Get the input slice first
    auto slice = this->get_output(name);

    // Set the value to use assignment
    slice = value;
}

void MultiResponseMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                          std::size_t start,
                                          std::size_t stop) const
{
    auto sliced_message = DCHECK_NOTNULL(std::dynamic_pointer_cast<MultiResponseMessage>(new_message));

    sliced_message->offset = start;
    sliced_message->count  = stop - start;

    // Currently our output lengths should always match mess_count, and even if they didn't we wouldn't have any way
    // to associate rows in the output with rows in the dataframe. Note on the input side we have the seq_ids array
    // to but we don't have any equivelant for the output.
    DCHECK(this->count == this->mess_count)
        << "Number of rows in response output does not match number of messages in DF";

    // Pass onto the base
    DerivedMultiMessage::get_slice_impl(new_message, start, stop);
}

void MultiResponseMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                            const std::vector<std::pair<size_t, size_t>> &ranges,
                                            size_t num_selected_rows) const
{
    auto copied_message = DCHECK_NOTNULL(std::dynamic_pointer_cast<MultiResponseMessage>(new_message));
    DerivedMultiMessage::copy_ranges_impl(copied_message, ranges, num_selected_rows);

    copied_message->offset = 0;
    copied_message->count  = num_selected_rows;
    copied_message->memory = copy_output_ranges(ranges, num_selected_rows);
}

std::shared_ptr<ResponseMemory> MultiResponseMessage::copy_output_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    auto offset_ranges = apply_offset_to_ranges(offset, ranges);
    auto tensors       = memory->copy_tensor_ranges(offset_ranges, num_selected_rows);
    return std::make_shared<ResponseMemory>(num_selected_rows, std::move(tensors));
}

/****** MultiResponseMessageInterfaceProxy *************************/
std::shared_ptr<MultiResponseMessage> MultiResponseMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                                               cudf::size_type mess_offset,
                                                                               cudf::size_type mess_count,
                                                                               std::shared_ptr<ResponseMemory> memory,
                                                                               cudf::size_type offset,
                                                                               cudf::size_type count)
{
    return std::make_shared<MultiResponseMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::ResponseMemory> MultiResponseMessageInterfaceProxy::memory(MultiResponseMessage &self)
{
    return self.memory;
}

std::size_t MultiResponseMessageInterfaceProxy::offset(MultiResponseMessage &self)
{
    return self.offset;
}

std::size_t MultiResponseMessageInterfaceProxy::count(MultiResponseMessage &self)
{
    return self.count;
}

pybind11::object MultiResponseMessageInterfaceProxy::get_output(MultiResponseMessage &self, const std::string &name)
{
    auto tensor = self.get_output(name);

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
