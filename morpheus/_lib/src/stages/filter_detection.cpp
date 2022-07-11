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

#include <morpheus/messages/memory/response_memory_probs.hpp>
#include <morpheus/stages/filter_detection.hpp>
#include <morpheus/utilities/matx_util.hpp>
#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <glog/logging.h>

#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <utility>

namespace morpheus {

std::vector<TensorIndex> get_element_stride(const std::vector<std::size_t> &stride)
{
    // Depending on the input the stride is given in bytes or elements,
    // divide the stride elements by the smallest item to ensure tensor_stride is defined in
    // terms of elements
    std::vector<TensorIndex> tensor_stride(stride.size());
    auto min_stride = std::min_element(stride.cbegin(), stride.cend());

    std::transform(stride.cbegin(),
                   stride.cend(),
                   tensor_stride.begin(),
                   std::bind(std::divides<>(), std::placeholders::_1, *min_stride));
    return tensor_stride;
}

std::shared_ptr<MultiResponseProbsMessage> combine_slices(
    std::size_t num_selected_rows, std::vector<std::shared_ptr<MultiResponseProbsMessage>> &&message_slices)
{
    // TODO: Move this to a method on the message class called "copy_ranges" and don't be lazy and make multible message
    // slices
    CHECK(num_selected_rows > 0);
    std::vector<cudf::table_view> sliced_views;
    std::map<std::string, TensorObject> outputs;
    std::map<std::string, uint8_t *> output_offsets;
    std::vector<std::string> column_names = message_slices.front()->get_meta().get_column_names();

    // peak at the first slice to determine what our outputs look like
    for (const auto &p : message_slices.front()->memory->outputs)
    {
        // using get_output as it applies the offsets to the output
        const auto &input_tensor = message_slices.front()->get_output(p.first);
        const auto num_columns   = input_tensor.shape(1);
        auto output_buffer       = std::make_shared<rmm::device_buffer>(
            num_selected_rows * num_columns * input_tensor.dtype_size(), rmm::cuda_stream_per_thread);

        output_offsets.insert(std::pair{p.first, static_cast<uint8_t *>(output_buffer->data())});

        outputs.insert(std::pair{
            p.first,
            Tensor::create(
                output_buffer, input_tensor.dtype(), {static_cast<TensorIndex>(num_selected_rows), num_columns}, {})});
    }

    // populate the outut buffers by
    for (const auto &msg : message_slices)
    {
        sliced_views.emplace_back(msg->get_meta().get_view());
        const auto slice_start_row = msg->offset;
        const auto slice_end_row   = msg->offset + msg->count;

        for (const auto &p : msg->memory->outputs)
        {
            // using get_output as it applies the offsets to the output
            const auto &input_tensor            = msg->get_output(p.first);
            const auto &shape                   = input_tensor.get_shape();
            const std::size_t num_input_rows    = shape[0];
            const std::size_t num_input_columns = shape[1];
            const auto stride                   = get_element_stride(input_tensor.get_stride());
            const auto row_stride               = stride[0];
            const auto item_size                = input_tensor.dtype().item_size();
            CHECK_EQ(num_input_rows, msg->count);

            if (row_stride == 1)
            {
                // column major just use cudaMemcpy
                SRF_CHECK_CUDA(cudaMemcpy(
                    output_offsets[p.first], input_tensor.data(), input_tensor.bytes(), cudaMemcpyDeviceToDevice));
            }
            else
            {
                SRF_CHECK_CUDA(cudaMemcpy2D(output_offsets[p.first],
                                            item_size,
                                            input_tensor.data(),
                                            row_stride * item_size,
                                            item_size,
                                            num_input_rows,
                                            cudaMemcpyDeviceToDevice));
            }

            output_offsets[p.first] += input_tensor.bytes();
        }
    }

    // concatenate returns a copy
    cudf::io::table_metadata metadata{};
    column_names.insert(column_names.begin(), std::string());  // cudf id col
    metadata.column_names               = std::move(column_names);
    cudf::io::table_with_metadata table = {cudf::concatenate(sliced_views, rmm::mr::get_current_device_resource()),
                                           std::move(metadata)};

    auto msg_meta = MessageMeta::create_from_cpp(std::move(table), 1);
    auto mem      = std::make_shared<ResponseMemoryProbs>(num_selected_rows, std::move(outputs));

    return std::make_shared<MultiResponseProbsMessage>(
        std::move(msg_meta), 0, num_selected_rows, std::move(mem), 0, num_selected_rows);
}

// Component public implementations
// ************ FilterDetectionStage **************************** //
FilterDetectionsStage::FilterDetectionsStage(float threshold) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_threshold(threshold)
{}

FilterDetectionsStage::subscribe_fn_t FilterDetectionsStage::build_operator()
{
    return
        [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
            return input.subscribe(rxcpp::make_observer<sink_type_t>(
                [this, &output](sink_type_t x) {
                    const auto &probs  = x->get_probs();
                    const auto &shape  = probs.get_shape();
                    const auto &stride = probs.get_stride();

                    CHECK(probs.rank() == 2)
                        << "C++ impl of the FilterDetectionsStage currently only supports two dimensional arrays";

                    const std::size_t num_rows    = shape[0];
                    const std::size_t num_columns = shape[1];

                    // A bit ugly, but we cant get access to the rmm::device_buffer here. So make a copy
                    auto tmp_buffer = std::make_shared<rmm::device_buffer>(probs.count() * probs.dtype_size(),
                                                                           rmm::cuda_stream_per_thread);

                    SRF_CHECK_CUDA(
                        cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

                    auto tensor_stride = get_element_stride(stride);

                    // Now call the threshold function
                    auto thresh_bool_buffer =
                        MatxUtil::threshold(DevMemInfo{probs.count(), probs.dtype().type_id(), tmp_buffer, 0},
                                            num_rows,
                                            num_columns,
                                            tensor_stride,
                                            m_threshold,
                                            true);
                    std::vector<uint8_t> host_bool_values(num_rows);

                    // Copy bools back to host
                    SRF_CHECK_CUDA(cudaMemcpy(host_bool_values.data(),
                                              thresh_bool_buffer->data(),
                                              thresh_bool_buffer->size(),
                                              cudaMemcpyDeviceToHost));

                    // We aren't sending these, but making use of the fact that the message classes
                    // already know how to slice the data frame and all of the outputs for us.
                    std::vector<std::shared_ptr<MultiResponseProbsMessage>> message_slices;
                    std::size_t num_selected_rows = 0;

                    // We are slicing by rows, using num_rows as our marker for undefined
                    std::size_t slice_start = num_rows;
                    for (std::size_t row = 0; row < num_rows; ++row)
                    {
                        bool above_threshold = host_bool_values[row];

                        if (above_threshold && slice_start == num_rows)
                        {
                            slice_start = row;
                        }
                        else if (!above_threshold && slice_start != num_rows)
                        {
                            message_slices.emplace_back(std::move(x->get_slice(slice_start, row)));
                            num_selected_rows += (row - slice_start);
                            slice_start = num_rows;
                        }
                    }

                    if (slice_start != num_rows)
                    {
                        // Last row was above the threshold
                        message_slices.emplace_back(std::move(x->get_slice(slice_start, num_rows)));
                        num_selected_rows += (num_rows - slice_start);
                    }

                    if (num_selected_rows > 0)
                    {
                        output.on_next(std::move(combine_slices(num_selected_rows, std::move(message_slices))));
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
}

// ************ FilterDetectionStageInterfaceProxy ************* //
std::shared_ptr<srf::segment::Object<FilterDetectionsStage>> FilterDetectionStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name, float threshold)
{
    auto stage = builder.construct_object<FilterDetectionsStage>(name, threshold);

    return stage;
}
}  // namespace morpheus
