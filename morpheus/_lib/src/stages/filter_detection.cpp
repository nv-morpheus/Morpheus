/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/stages/filter_detection.hpp>

#include <morpheus/utilities/matx_util.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>

namespace morpheus {
// Component public implementations
// ************ FilterDetectionStage **************************** //
FilterDetectionsStage::FilterDetectionsStage(const neo::Segment &parent, const std::string &name, float threshold) :
  neo::SegmentObject(parent, name),
  PythonNode(parent, name, build_operator()),
  m_threshold(threshold)
{}

FilterDetectionsStage::operator_fn_t FilterDetectionsStage::build_operator()
{
    return
        [this](neo::Observable<reader_type_t> &input, neo::Subscriber<writer_type_t> &output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t &&x) {
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

                    NEO_CHECK_CUDA(
                        cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

                    // Depending on the input the stride is given in bytes or elements,
                    // divide the stride elements by the smallest item to ensure tensor_stride is defined in
                    // terms of elements
                    std::vector<neo::TensorIndex> tensor_stride(stride.size());
                    auto min_stride = std::min_element(stride.cbegin(), stride.cend());

                    std::transform(stride.cbegin(),
                                   stride.cend(),
                                   tensor_stride.begin(),
                                   std::bind(std::divides<>(), std::placeholders::_1, *min_stride));

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
                    NEO_CHECK_CUDA(cudaMemcpy(host_bool_values.data(),
                                              thresh_bool_buffer->data(),
                                              thresh_bool_buffer->size(),
                                              cudaMemcpyDeviceToHost));

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
                            output.on_next(x->get_slice(slice_start, row));
                            slice_start = num_rows;
                        }
                    }

                    if (slice_start != num_rows)
                    {
                        // Last row was above the threshold
                        output.on_next(x->get_slice(slice_start, num_rows));
                    }
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
}

// ************ FilterDetectionStageInterfaceProxy ************* //
std::shared_ptr<FilterDetectionsStage> FilterDetectionStageInterfaceProxy::init(neo::Segment &parent,
                                                                                const std::string &name,
                                                                                float threshold)
{
    auto stage = std::make_shared<FilterDetectionsStage>(parent, name, threshold);

    parent.register_node<FilterDetectionsStage>(stage);

    return stage;
}
}  // namespace morpheus
