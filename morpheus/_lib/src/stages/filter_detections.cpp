/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/filter_detections.hpp"  // IWYU pragma: accosiated

#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "pymrc/node.hpp"

#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_tensor.hpp"
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DataType
#include "morpheus/objects/memory_descriptor.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/types.hpp"                  // for RangeType
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>       // for CHECK, CHECK_NE
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>  // for device_buffer
#include <rmm/mr/device/per_device_resource.hpp>
#include <spdlog/sinks/sink.h>

#include <cstddef>
#include <cstdint>  // for uint8_t
#include <exception>
#include <functional>
#include <memory>
#include <ostream>  // needed for glog
#include <string>
#include <utility>  // for pair
#include <vector>
// IWYU thinks we need ext/new_allocator.h for size_t for some reason
// IWYU pragma: no_include <ext/new_allocator.h>

namespace morpheus {

// Component public implementations
// ************ FilterDetectionStage **************************** //
template <typename InputT, typename OutputT>
FilterDetectionsStage<InputT, OutputT>::FilterDetectionsStage(float threshold,
                                                              bool copy,
                                                              FilterSource filter_source,
                                                              std::string field_name) :
  base_t(rxcpp::operators::map([this](sink_type_t x) {
      return this->on_data(std::move(x));
  })),
  m_threshold(threshold),
  m_copy(copy),
  m_filter_source(filter_source),
  m_field_name(std::move(field_name))
{
    CHECK(m_filter_source != FilterSource::Auto);  // The python stage should determine this
}

template <typename InputT, typename OutputT>
DevMemInfo FilterDetectionsStage<InputT, OutputT>::get_tensor_filter_source(const sink_type_t& x)
{
    TensorObject filter_source;
    if constexpr (std::is_same_v<sink_type_t, const std::shared_ptr<MultiMessage>&>)
    {
        // The pipeline build will check to ensure that our input is a MultiResponseMessage
        filter_source = std::static_pointer_cast<MultiTensorMessage>(x)->get_tensor(m_field_name);
    }
    else if constexpr (std::is_same_v<sink_type_t, const std::shared_ptr<ControlMessage>&>)
    {
        filter_source = x->tensors()->get_tensor(m_field_name);
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"FilterDetectionStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }

    CHECK(filter_source.rank() > 0 && filter_source.rank() <= 2)
        << "C++ impl of the FilterDetectionsStage currently only supports one and two dimensional "
           "arrays";

    // Depending on the input the stride is given in bytes or elements, convert to elements
    auto stride = TensorUtils::get_element_stride(filter_source.get_stride());
    return {filter_source.data(), filter_source.dtype(), filter_source.get_memory(), filter_source.get_shape(), stride};
}

template <typename InputT, typename OutputT>
DevMemInfo FilterDetectionsStage<InputT, OutputT>::get_column_filter_source(const sink_type_t& x)
{
    TableInfo table_info;
    if constexpr (std::is_same_v<sink_type_t, const std::shared_ptr<MultiMessage>&>)
    {
        table_info = x->get_meta(m_field_name);
    }
    else if constexpr (std::is_same_v<sink_type_t, const std::shared_ptr<ControlMessage>&>)
    {
        table_info = x->payload()->get_data(m_field_name);
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"FilterDetectionsStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }

    // since we only asked for one column, we know its the first
    const auto& col = table_info.get_column(0);
    auto dtype      = DType::from_cudf(col.type().id());
    auto num_rows   = col.size();
    auto data =
        const_cast<uint8_t*>(static_cast<const uint8_t*>(col.head<uint8_t>() + col.offset() * dtype.item_size()));

    return {
        data,
        std::move(dtype),
        std::make_shared<MemoryDescriptor>(rmm::cuda_stream_per_thread, rmm::mr::get_current_device_resource()),
        {num_rows, 1},
        {1, 0},
    };
}

template <typename InputT, typename OutputT>
FilterDetectionsStage<InputT, OutputT>::source_type_t FilterDetectionsStage<InputT, OutputT>::on_data(sink_type_t x)
{
    std::function<DevMemInfo(const sink_type_t& x)> get_filter_source;

    if (m_filter_source == FilterSource::TENSOR)
    {
        get_filter_source = [this](auto x) {
            return get_tensor_filter_source(x);
        };
    }
    else
    {
        get_filter_source = [this](auto x) {
            return get_column_filter_source(x);
        };
    }

    auto tmp_buffer = get_filter_source(x);

    const auto num_rows    = tmp_buffer.shape(0);
    const auto num_columns = tmp_buffer.shape(1);

    bool by_row = (num_columns > 1);

    // Now call the threshold function
    auto thresh_bool_buffer = MatxUtil::threshold(tmp_buffer, m_threshold, by_row);

    std::vector<uint8_t> host_bool_values(num_rows);

    // Copy bools back to host
    MRC_CHECK_CUDA(cudaMemcpy(
        host_bool_values.data(), thresh_bool_buffer->data(), thresh_bool_buffer->size(), cudaMemcpyDeviceToHost));

    // Only used when m_copy is true
    std::vector<RangeType> selected_ranges;
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
                if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
                {
                    return x->get_slice(slice_start, row);
                }
                else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
                {
                    auto meta = x->payload();
                    x->payload(meta->get_slice(slice_start, row));
                    return x;
                }
                // sink_type_t not supported
                else
                {
                    std::string error_msg{"FilterDetectionsStage receives unsupported input type: " +
                                          std::string(typeid(x).name())};
                    LOG(ERROR) << error_msg;
                    throw std::runtime_error(error_msg);
                }
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
            if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
            {
                return x->get_slice(slice_start, num_rows);
            }
            else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
            {
                auto meta = x->payload();
                x->payload(meta->get_slice(slice_start, num_rows));
                return x;
            }
            // sink_type_t not supported
            else
            {
                std::string error_msg{"FilterDetectionsStage receives unsupported input type: " +
                                      std::string(typeid(x).name())};
                LOG(ERROR) << error_msg;
                throw std::runtime_error(error_msg);
            }
        }
    }

    // num_selected_rows will always be 0 when m_copy is false,
    // or when m_copy is true, but none of the rows matched the output
    if (num_selected_rows > 0)
    {
        DCHECK(m_copy);
        if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
        {
            return x->copy_ranges(selected_ranges, num_selected_rows);
        }
        else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
        {
            auto meta = x->payload();
            x->payload(meta->copy_ranges(selected_ranges));
            return x;
        }
        // sink_type_t not supported
        else
        {
            std::string error_msg{"FilterDetectionsStage receives unsupported input type: " +
                                  std::string(typeid(x).name())};
            LOG(ERROR) << error_msg;
            throw std::runtime_error(error_msg);
        }
    }
}

// ************ FilterDetectionStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<FilterDetectionsStageMM>> FilterDetectionStageInterfaceProxy::init_mm(
    mrc::segment::Builder& builder,
    const std::string& name,
    float threshold,
    bool copy,
    FilterSource filter_source,
    std::string field_name)
{
    auto stage = builder.construct_object<FilterDetectionsStageMM>(name, threshold, copy, filter_source, field_name);

    return stage;
}

std::shared_ptr<mrc::segment::Object<FilterDetectionsStageCM>> FilterDetectionStageInterfaceProxy::init_cm(
    mrc::segment::Builder& builder,
    const std::string& name,
    float threshold,
    bool copy,
    FilterSource filter_source,
    std::string field_name)
{
    auto stage = builder.construct_object<FilterDetectionsStageCM>(name, threshold, copy, filter_source, field_name);

    return stage;
}
}  // namespace morpheus
