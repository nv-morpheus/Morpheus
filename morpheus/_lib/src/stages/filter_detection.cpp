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

#include "morpheus/stages/filter_detection.hpp"  // IWYU pragma: accosiated

#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/filter_source.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"       // for TensorUtils::get_element_stride
#include "morpheus/utilities/type_util_detail.hpp"  // for DataType

#include <cuda_runtime.h>            // for cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost
#include <glog/logging.h>            // for CHECK
#include <mrc/cuda/common.hpp>       // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstddef>
#include <cstdint>  // for uint8_t
#include <exception>
#include <memory>
#include <ostream>  // needed for glog
#include <string>
#include <type_traits>  // for declval (indirectly via templates)
#include <utility>      // for pair
// IWYU thinks we need ext/new_allocator.h for size_t for some reason
// IWYU pragma: no_include <ext/new_allocator.h>

namespace morpheus {

// Component public implementations
// ************ FilterDetectionStage **************************** //
FilterDetectionsStage::FilterDetectionsStage(float threshold,
                                             bool copy,
                                             FilterSource operate_on,
                                             std::string field_name) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_threshold(threshold),
  m_copy(copy),
  m_operate_on(operate_on),
  m_field_name(std::move(field_name))
{}

FilterDetectionsStage::subscribe_fn_t FilterDetectionsStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                const auto& filter_source = x->get_output(m_field_name);
                const auto& shape         = filter_source.get_shape();
                const auto& stride        = filter_source.get_stride();

                CHECK(filter_source.rank() == 2)
                    << "C++ impl of the FilterDetectionsStage currently only supports two dimensional arrays";

                const std::size_t num_rows    = shape[0];
                const std::size_t num_columns = shape[1];

                // A bit ugly, but we cant get access to the rmm::device_buffer here. So make a copy
                auto tmp_buffer = std::make_shared<rmm::device_buffer>(
                    filter_source.count() * filter_source.dtype_size(), rmm::cuda_stream_per_thread);

                MRC_CHECK_CUDA(
                    cudaMemcpy(tmp_buffer->data(), filter_source.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

                // Depending on the input the stride is given in bytes or elements, convert to elements
                auto tensor_stride = TensorUtils::get_element_stride<TensorIndex, std::size_t>(stride);

                // Now call the threshold function
                auto thresh_bool_buffer = MatxUtil::threshold(
                    DevMemInfo{filter_source.count(), filter_source.dtype().type_id(), tmp_buffer, 0},
                    num_rows,
                    num_columns,
                    tensor_stride,
                    m_threshold,
                    true);
                std::vector<uint8_t> host_bool_values(num_rows);

                // Copy bools back to host
                MRC_CHECK_CUDA(cudaMemcpy(host_bool_values.data(),
                                          thresh_bool_buffer->data(),
                                          thresh_bool_buffer->size(),
                                          cudaMemcpyDeviceToHost));

                // Only used when m_copy is true
                std::vector<std::pair<std::size_t, std::size_t>> selected_ranges;
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
                        if (m_copy)
                        {
                            selected_ranges.emplace_back(std::pair{slice_start, row});
                            num_selected_rows += (row - slice_start);
                        }
                        else
                        {
                            output.on_next(x->get_slice(slice_start, row));
                        }

                        slice_start = num_rows;
                    }
                }

                if (slice_start != num_rows)
                {
                    // Last row was above the threshold
                    if (m_copy)
                    {
                        selected_ranges.emplace_back(std::pair{slice_start, num_rows});
                        num_selected_rows += (num_rows - slice_start);
                    }
                    else
                    {
                        output.on_next(x->get_slice(slice_start, num_rows));
                    }
                }

                // num_selected_rows will always be 0 when m_copy is false,
                // or when m_copy is true, but none of the rows matched the output
                if (num_selected_rows > 0)
                {
                    DCHECK(m_copy);
                    output.on_next(x->copy_ranges(selected_ranges, num_selected_rows));
                }
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ FilterDetectionStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<FilterDetectionsStage>> FilterDetectionStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    float threshold,
    bool copy,
    FilterSource operate_on,
    std::string field_name)
{
    auto stage = builder.construct_object<FilterDetectionsStage>(name, threshold, copy, operate_on, field_name);

    return stage;
}
}  // namespace morpheus
