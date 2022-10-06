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

#include "morpheus/messages/multi_inference.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory::tensor_map_t
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <cstdint>  // for int32_t
#include <memory>
#include <ostream>  // needed for logging
#include <string>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** <MultiInferenceMessage>****************************************/
MultiInferenceMessage::MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                             std::size_t mess_offset,
                                             std::size_t mess_count,
                                             std::shared_ptr<morpheus::InferenceMemory> memory,
                                             std::size_t offset,
                                             std::size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count),
  memory(std::move(memory)),
  offset(offset),
  count(count)
{}

const TensorObject MultiInferenceMessage::get_input(const std::string &name) const
{
    CHECK(this->memory->has_input(name)) << "Cound not find input: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->tensors[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->tensors[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const void MultiInferenceMessage::set_input(const std::string &name, const TensorObject &value)
{
    // Get the input slice first
    auto slice = this->get_input(name);

    // Set the value to use assignment
    slice = value;
}

void MultiInferenceMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                           std::size_t start,
                                           std::size_t stop) const
{
    auto sliced_message = DCHECK_NOTNULL(std::dynamic_pointer_cast<MultiInferenceMessage>(new_message));

    sliced_message->offset = start;
    sliced_message->count  = stop - start;

    // If we have more inference rows than message rows, we need to use the seq_ids to figure out the slicing. This
    // will be slow and should be avoided at all costs
    if (this->count != this->mess_count && this->memory->has_input("seq_ids"))
    {
        auto seq_ids = this->get_input("seq_ids");

        // Determine the new start and stop before passing onto the base
        start = seq_ids.read_element<int32_t>({(TensorIndex)start, 0});
        stop  = seq_ids.read_element<int32_t>({(TensorIndex)stop - 1, 0}) + 1;
    }

    // Pass onto the base
    DerivedMultiMessage::get_slice_impl(new_message, start, stop);
}

void MultiInferenceMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                             const std::vector<std::pair<size_t, size_t>> &ranges,
                                             size_t num_selected_rows) const
{
    auto copied_message = DCHECK_NOTNULL(std::dynamic_pointer_cast<MultiInferenceMessage>(new_message));
    DerivedMultiMessage::copy_ranges_impl(copied_message, ranges, num_selected_rows);

    copied_message->offset = 0;
    copied_message->count  = num_selected_rows;
    copied_message->memory = copy_input_ranges(ranges, num_selected_rows);
}

std::shared_ptr<InferenceMemory> MultiInferenceMessage::copy_input_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    auto offset_ranges = apply_offset_to_ranges(offset, ranges);
    auto tensors       = memory->copy_tensor_ranges(offset_ranges, num_selected_rows);
    return std::make_shared<InferenceMemory>(num_selected_rows, std::move(tensors));
}

/****** <MultiInferenceMessage>InterfaceProxy *************************/
std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<InferenceMemory> memory,
    cudf::size_type offset,
    cudf::size_type count)
{
    return std::make_shared<MultiInferenceMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::InferenceMemory> MultiInferenceMessageInterfaceProxy::memory(MultiInferenceMessage &self)
{
    return self.memory;
}

std::size_t MultiInferenceMessageInterfaceProxy::offset(MultiInferenceMessage &self)
{
    return self.offset;
}

std::size_t MultiInferenceMessageInterfaceProxy::count(MultiInferenceMessage &self)
{
    return self.count;
}

pybind11::object MultiInferenceMessageInterfaceProxy::get_input(MultiInferenceMessage &self, const std::string &name)
{
    const auto &py_tensor = CupyUtil::tensor_to_cupy(self.get_input(name));
    return py_tensor;
}

std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::get_slice(MultiInferenceMessage &self,
                                                                                      std::size_t start,
                                                                                      std::size_t stop)
{
    return self.get_slice(start, stop);
}
}  // namespace morpheus
