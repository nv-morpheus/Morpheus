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

#include <morpheus/messages/multi_response.hpp>

#include <morpheus/messages/memory/response_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>
#include <morpheus/utilities/tensor_util.hpp>

#include <cudf/types.hpp>

#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include "morpheus/objects/tensor_object.hpp"

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseMessage****************************************/
MultiResponseMessage::MultiResponseMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                           std::size_t mess_offset,
                                           std::size_t mess_count,
                                           std::shared_ptr<ResponseMemory> memory,
                                           std::size_t offset,
                                           std::size_t count) :
  MultiMessage(meta, mess_offset, mess_count),
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
        return this->memory->outputs[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const TensorObject MultiResponseMessage::get_output(const std::string &name) const
{
    CHECK(this->memory->has_output(name)) << "Could not find output: " << name;

    // check if we are getting the entire input
    if (this->offset == 0 && this->count == this->memory->count)
    {
        return this->memory->outputs[name];
    }

    // TODO(MDD): This really needs to return the slice of the tensor
    return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                             {static_cast<cudf::size_type>(this->offset + this->count), -1});
}

const void MultiResponseMessage::set_output(const std::string &name, const TensorObject &value)
{
    // Get the input slice first
    auto slice = this->get_output(name);

    // Set the value to use assignment
    slice = value;
}

std::shared_ptr<MultiResponseMessage> MultiResponseMessage::get_slice(std::size_t start, std::size_t stop) const
{
    // This can only cast down
    return std::static_pointer_cast<MultiResponseMessage>(this->internal_get_slice(start, stop));
}

std::shared_ptr<MultiMessage> MultiResponseMessage::internal_get_slice(std::size_t start, std::size_t stop) const
{
    CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

    auto mess_start = this->mess_offset + start;
    auto mess_stop  = this->mess_offset + stop;

    return std::make_shared<MultiResponseMessage>(
        this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
}

std::shared_ptr<MultiResponseMessage> MultiResponseMessage::copy_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    return std::static_pointer_cast<MultiResponseMessage>(this->internal_copy_ranges(ranges, num_selected_rows));
}

std::shared_ptr<MultiMessage> MultiResponseMessage::internal_copy_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    auto msg_meta = copy_meta_ranges(ranges);
    auto mem      = copy_output_ranges(ranges, num_selected_rows);
    return std::make_shared<MultiResponseMessage>(msg_meta, 0, num_selected_rows, mem, 0, num_selected_rows);
}

std::shared_ptr<ResponseMemory> MultiResponseMessage::copy_output_ranges(
    const std::vector<std::pair<size_t, size_t>> &ranges, size_t num_selected_rows) const
{
    std::map<std::string, TensorObject> output_tensors;
    for (const auto &mem_output : memory->outputs)
    {
        // A little confusing here, but the response outputs are the inputs for this method
        const std::string &output_name   = mem_output.first;
        const TensorObject &input_tensor = mem_output.second;

        const auto dtype     = input_tensor.dtype();
        const auto item_size = dtype.item_size();
        const std::size_t num_columns =
            input_tensor.get_shape()[1];  // number of columns should be the same on the input & output

        const auto stride     = TensorUtils::get_element_stride(input_tensor.get_stride());
        const auto row_stride = stride[0];

        auto output_buffer = std::make_shared<rmm::device_buffer>(num_selected_rows * num_columns * item_size,
                                                                  rmm::cuda_stream_per_thread);

        auto output_offset = static_cast<uint8_t *>(output_buffer->data());

        for (const auto &range : ranges)
        {
            const auto &sliced_input_tensor =
                input_tensor.slice({static_cast<cudf::size_type>(this->offset + range.first), 0},
                                   {static_cast<cudf::size_type>(this->offset + range.second), -1});
            const std::size_t num_input_rows = sliced_input_tensor.get_shape()[0];
            CHECK_EQ(num_input_rows, range.second - range.first);

            if (row_stride == 1)
            {
                // column major just use cudaMemcpy
                SRF_CHECK_CUDA(cudaMemcpy(
                    output_offset, sliced_input_tensor.data(), sliced_input_tensor.bytes(), cudaMemcpyDeviceToDevice));
            }
            else
            {
                SRF_CHECK_CUDA(cudaMemcpy2D(output_offset,
                                            item_size,
                                            sliced_input_tensor.data(),
                                            row_stride * item_size,
                                            item_size,
                                            num_input_rows,
                                            cudaMemcpyDeviceToDevice));
            }

            output_offset += sliced_input_tensor.bytes();
        }

        output_tensors.insert(std::pair{
            output_name,
            Tensor::create(output_buffer,
                           dtype,
                           {static_cast<TensorIndex>(num_selected_rows), static_cast<TensorIndex>(num_columns)},
                           {})});
    }

    return std::make_shared<ResponseMemory>(num_selected_rows, std::move(output_tensors));
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
