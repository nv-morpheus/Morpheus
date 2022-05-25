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

#include <morpheus/messages/multi_inference.hpp>

#include <morpheus/messages/memory/inference_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>

#include <pybind11/pytypes.h>
#include <cudf/types.hpp>

#include <memory>
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
  MultiMessage(meta, mess_offset, mess_count),
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
        return this->memory->inputs[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->inputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                            {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const void MultiInferenceMessage::set_input(const std::string &name, const TensorObject &value)
{
    // Get the input slice first
    auto slice = this->get_input(name);

    // Set the value to use assignment
    slice = value;
}

std::shared_ptr<MultiInferenceMessage> MultiInferenceMessage::get_slice(std::size_t start, std::size_t stop) const
{
    // This can only cast down
    return std::static_pointer_cast<MultiInferenceMessage>(this->internal_get_slice(start, stop));
}

std::shared_ptr<MultiMessage> MultiInferenceMessage::internal_get_slice(std::size_t start, std::size_t stop) const
{
    CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

    auto mess_start = this->mess_offset + start;
    auto mess_stop  = this->mess_offset + stop;

    // If we have more inference rows than message rows, we need to use the seq_ids to figure out the slicing. This
    // will be slow and should be avoided at all costs
    if (this->memory->has_input("seq_ids") && this->count != this->mess_count)
    {
        auto seq_ids = this->get_input("seq_ids");

        // Convert to MatX to access elements
        mess_start = this->mess_offset + seq_ids.read_element<int32_t>({(TensorIndex)start, 0});
        mess_stop  = this->mess_offset + seq_ids.read_element<int32_t>({(TensorIndex)stop - 1, 0}) + 1;
    }

    return std::make_shared<MultiInferenceMessage>(
        this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
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

    //  //  Need to get just our portion. TODO(MDD): THis should be handled in get_input
    //  py::object sliced = py_tensor[py::make_tuple(
    //      py::slice(py::int_(self.offset), py::int_(self.offset + self.count), py::none()),
    //      py::slice(py::none(), py::none(), py::none()))];

    return py_tensor;
}

std::shared_ptr<MultiInferenceMessage> MultiInferenceMessageInterfaceProxy::get_slice(MultiInferenceMessage &self,
                                                                                      std::size_t start,
                                                                                      std::size_t stop)
{
    // py::object seq_ids = CupyUtil::tensor_to_cupy(self.get_input("seq_ids"), m);

    // int mess_start = seq_ids[py::make_tuple(start, 0)].attr("item")().cast<int>();
    // int mess_stop  = seq_ids[py::make_tuple(stop - 1, 0)].attr("item")().cast<int>() + 1;

    // return std::make_shared<MultiInferenceMessage>(
    //     self.meta, mess_start, mess_stop - mess_start, self.memory, start, stop - start);
    return self.get_slice(start, stop);
}
}  // namespace morpheus
