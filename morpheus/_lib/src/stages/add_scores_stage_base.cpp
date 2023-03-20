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

#include "morpheus/stages/add_scores_stage_base.hpp"

#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DType
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/types.hpp"                  // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/string_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyDeviceToDevice
#include <glog/logging.h>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <operators/rx-map.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstddef>
#include <iterator>
#include <memory>
#include <ostream>  // needed for logging
#include <utility>  // for move
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {

// Component public implementations
// ************ AddClassificationStage **************************** //
AddScoresStageBase::AddScoresStageBase(std::map<std::size_t, std::string> idx2label, std::optional<float> threshold) :
  PythonNode(),
  m_idx2label(std::move(idx2label)),
  m_threshold(threshold),
  m_min_col_count(m_idx2label.rbegin()->first)  // Ordered map's largest key will be the last entry
{
    this->pipe(rxcpp::operators::map([this](sink_type_t x) { return this->on_data(std::move(x)); }));
}

AddScoresStageBase::source_type_t AddScoresStageBase::on_data(sink_type_t x)
{
    const auto& probs = x->get_probs_tensor();
    const auto& shape = probs.get_shape();

    // Depending on the input the stride is given in bytes or elements, convert to elements
    auto stride = TensorUtils::get_element_stride(probs.get_stride());

    CHECK(shape.size() == 2 && shape[1] > m_min_col_count)
        << "Model output did not contain enough columns to fufill the requested labels. Label "
           "indexes: "
        << StringUtil::map_to_str(m_idx2label.begin(), m_idx2label.end()) << ", Model output columns: " << shape[1];

    const auto num_rows    = shape[0];
    const auto num_columns = shape[1];

    TensorObject output_tensor;

    if (m_threshold.has_value())
    {
        // A bit ugly, but we cant get access to the rmm::device_buffer here. So make a copy
        auto tmp_buffer = std::make_shared<rmm::device_buffer>(probs.bytes(), rmm::cuda_stream_per_thread);

        MRC_CHECK_CUDA(cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

        // Now call the threshold function
        auto thresh_bool_buffer =
            MatxUtil::threshold(DevMemInfo{tmp_buffer, probs.dtype(), shape, stride}, *m_threshold, false);

        output_tensor = Tensor::create(thresh_bool_buffer, DType::create<bool>(), shape, stride);
    }
    else
    {
        output_tensor = std::move(probs);
    }

    std::vector<std::string> columns(m_idx2label.size());
    std::vector<TensorObject> tensors(m_idx2label.size());

    std::size_t i = 0;
    for (const auto& [column_num, column_name] : m_idx2label)
    {
        columns[i] = column_name;
        tensors[i] = output_tensor.slice({0, static_cast<TensorIndex>(column_num)},
                                         {num_rows, static_cast<TensorIndex>(column_num + 1)});

        ++i;
    }

    x->set_meta(columns, tensors);

    return x;
}

}  // namespace morpheus
