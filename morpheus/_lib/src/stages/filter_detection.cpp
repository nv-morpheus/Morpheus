/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DataType
#include "morpheus/objects/memory_descriptor.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>            // for CHECK, CHECK_NE
#include <mrc/cuda/common.hpp>       // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstddef>
#include <cstdint>  // for uint8_t
#include <exception>
#include <functional>
#include <memory>
#include <ostream>  // needed for glog
#include <string>
#include <utility>  // for pair
// IWYU thinks we need ext/new_allocator.h for size_t for some reason
// IWYU pragma: no_include <ext/new_allocator.h>

namespace morpheus {

// Component public implementations
// ************ FilterDetectionStage **************************** //
FilterDetectionsStage::FilterDetectionsStage(float threshold,
                                             bool copy,
                                             FilterSource filter_source,
                                             std::string field_name) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_threshold(threshold),
  m_copy(copy),
  m_filter_source(filter_source),
  m_field_name(std::move(field_name))
{
    CHECK(m_filter_source != FilterSource::Auto);  // The python stage should determine this
}

DevMemInfo FilterDetectionsStage::get_tensor_filter_source(const std::shared_ptr<morpheus::MultiMessage>& x)
{
    // The pipeline build will check to ensure that our input is a MultiResponseMessage
    const auto& filter_source = std::static_pointer_cast<morpheus::MultiTensorMessage>(x)->get_tensor(m_field_name);
    CHECK(filter_source.rank() > 0 && filter_source.rank() <= 2)
        << "C++ impl of the FilterDetectionsStage currently only supports one and two dimensional "
           "arrays";

    // Depending on the input the stride is given in bytes or elements, convert to elements
    auto stride = morpheus::TensorUtils::get_element_stride<std::size_t, std::size_t>(filter_source.get_stride());
    return {filter_source.data(), filter_source.dtype(), filter_source.get_memory(), filter_source.get_shape(), stride};
}

DevMemInfo FilterDetectionsStage::get_column_filter_source(const std::shared_ptr<morpheus::MultiMessage>& x)
{
    auto table_info = x->get_meta(m_field_name);

    // since we only asked for one column, we know its the first
    const auto& col = table_info.get_column(0);
    auto dtype      = morpheus::DType::from_cudf(col.type().id());
    auto num_rows   = static_cast<std::size_t>(col.size());
    auto data =
        const_cast<uint8_t*>(static_cast<const uint8_t*>(col.head<uint8_t>() + col.offset() * dtype.item_size()));

    return {
        data,
        std::move(dtype),
        std::make_shared<MemoryDescriptor>(),
        {num_rows, 1},
        {1, 0},
    };
}

FilterDetectionsStage::subscribe_fn_t FilterDetectionsStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        std::function<DevMemInfo(const std::shared_ptr<morpheus::MultiMessage>& x)> get_filter_source;

        if (m_filter_source == FilterSource::TENSOR)
        {
            get_filter_source = [this](auto x) { return get_tensor_filter_source(x); };
        }
        else
        {
            get_filter_source = [this](auto x) { return get_column_filter_source(x); };
        }

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, &get_filter_source](sink_type_t x) {
                auto tmp_buffer = get_filter_source(x);

                const auto num_rows    = tmp_buffer.shape(0);
                const auto num_columns = tmp_buffer.shape(1);

                bool by_row = (num_columns > 1);

                // Now call the threshold function
                auto thresh_bool_buffer = MatxUtil::threshold(tmp_buffer, m_threshold, by_row);

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
    FilterSource filter_source,
    std::string field_name)
{
    auto stage = builder.construct_object<FilterDetectionsStage>(name, threshold, copy, filter_source, field_name);

    return stage;
}
}  // namespace morpheus
