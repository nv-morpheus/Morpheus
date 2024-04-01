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

#pragma once

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/multi_response.hpp"  // for MultiResponseMessage
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"

#include <boost/fiber/context.hpp>
#include <boost/fiber/future/future.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/types.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** AddClassificationStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief Base class for both `AddScoresStage` and `AddClassificationStage`
 */
template <typename InputT, typename OutputT>
class AddScoresStageBase : public mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Add Classifications Stage object
     *
     * @param threshold : Threshold to consider true/false for each class
     * @param idx2label : Index to classification labels map
     */
    AddScoresStageBase(std::map<std::size_t, std::string> idx2label, std::optional<float> threshold);

    /**
     * Called every time a message is passed to this stage
     */
    source_type_t on_data(sink_type_t x);

  private:
    void on_multi_response_message(std::shared_ptr<MultiResponseMessage> x);
    void on_control_message(std::shared_ptr<ControlMessage> x);
    std::map<std::size_t, std::string> m_idx2label;
    std::optional<float> m_threshold;

    // The minimum number of columns needed to extract the label data
    std::size_t m_min_col_count;
};

template <typename InputT, typename OutputT>
AddScoresStageBase<InputT, OutputT>::AddScoresStageBase(std::map<std::size_t, std::string> idx2label,
                                                        std::optional<float> threshold) :
  base_t(),
  m_idx2label(std::move(idx2label)),
  m_threshold(threshold),
  m_min_col_count(m_idx2label.rbegin()->first)  // Ordered map's largest key will be the last entry
{
    this->pipe(rxcpp::operators::map([this](sink_type_t x) {
        return this->on_data(std::move(x));
    }));
}

template <typename InputT, typename OutputT>
AddScoresStageBase<InputT, OutputT>::source_type_t AddScoresStageBase<InputT, OutputT>::on_data(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiResponseMessage>>)
    {
        this->on_multi_response_message(x);
    }
    else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
    {
        this->on_control_message(x);
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"AddScoresStageBase receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
    return x;
}

template <typename InputT, typename OutputT>
void AddScoresStageBase<InputT, OutputT>::on_multi_response_message(std::shared_ptr<MultiResponseMessage> x)
{
    auto probs        = x->get_probs_tensor();
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
        auto thresh_bool_buffer = MatxUtil::threshold(
            {probs.data(), probs.dtype(), probs.get_memory(), probs.get_shape(), probs.get_stride()},
            *m_threshold,
            false);

        output_tensor.swap(Tensor::create(thresh_bool_buffer, DType::create<bool>(), shape, stride));
    }
    else
    {
        output_tensor.swap(std::move(probs));
    }

    std::vector<std::string> columns;
    std::vector<TensorObject> tensors;

    std::size_t i = 0;
    for (const auto& [column_num, column_name] : m_idx2label)
    {
        columns.push_back(column_name);
        tensors.emplace_back(output_tensor.slice({0, static_cast<TensorIndex>(column_num)},
                                                 {num_rows, static_cast<TensorIndex>(column_num + 1)}));

        ++i;
    }

    x->set_meta(columns, tensors);
}

template <typename InputT, typename OutputT>
void AddScoresStageBase<InputT, OutputT>::on_control_message(std::shared_ptr<ControlMessage> x)
{
    // The default of probs_tensor_name is "probs"
    auto probs        = x->tensors()->get_tensor("probs");
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
        auto thresh_bool_buffer = MatxUtil::threshold(
            {probs.data(), probs.dtype(), probs.get_memory(), probs.get_shape(), probs.get_stride()},
            *m_threshold,
            false);

        output_tensor.swap(Tensor::create(thresh_bool_buffer, DType::create<bool>(), shape, stride));
    }
    else
    {
        output_tensor.swap(std::move(probs));
    }

    std::vector<std::string> columns;
    std::vector<TensorObject> tensors;

    std::size_t i = 0;
    for (const auto& [column_num, column_name] : m_idx2label)
    {
        columns.push_back(column_name);
        tensors.emplace_back(output_tensor.slice({0, static_cast<TensorIndex>(column_num)},
                                                 {num_rows, static_cast<TensorIndex>(column_num + 1)}));

        ++i;
    }

    // A copy of MultiMessage::set_meta(const std::vector<std::string>& column_names, const
    // std::vector<TensorObject>& tensors)
    TableInfo sliced_table_meta;
    try
    {
        TableInfo table_meta     = x->payload()->get_info();
        auto table_meta_num_rows = table_meta.num_rows();
        sliced_table_meta        = table_meta.get_slice(0, table_meta_num_rows, columns);
    } catch (const std::runtime_error& e)
    {
        std::ostringstream err_msg;
        err_msg << e.what() << " Ensure that the stage that needs this column has populated the '_needed_columns' "
                << "attribute and that at least one stage in the current segment is using the PreallocatorMixin to "
                << "ensure all needed columns have been allocated.";
        throw std::runtime_error(err_msg.str());
    }
    for (std::size_t i = 0; i < tensors.size(); ++i)
    {
        const auto& cv            = sliced_table_meta.get_column(i);
        const auto table_type_id  = cv.type().id();
        const auto tensor_type    = DType(tensors[i].dtype());
        const auto tensor_type_id = tensor_type.cudf_type_id();
        const auto row_stride     = tensors[i].stride(0);

        CHECK(tensors[i].count() == cv.size() &&
              (table_type_id == tensor_type_id ||
               (table_type_id == cudf::type_id::BOOL8 && tensor_type_id == cudf::type_id::UINT8)));

        const auto item_size = tensors[i].dtype().item_size();

        // Dont use cv.data<>() here since that does not account for the size of each element
        auto data_start = const_cast<uint8_t*>(cv.head<uint8_t>()) + cv.offset() * item_size;

        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            MRC_CHECK_CUDA(cudaMemcpy(data_start, tensors[i].data(), tensors[i].bytes(), cudaMemcpyDeviceToDevice));
        }
        else
        {
            MRC_CHECK_CUDA(cudaMemcpy2D(data_start,
                                        item_size,
                                        tensors[i].data(),
                                        row_stride * item_size,
                                        item_size,
                                        cv.size(),
                                        cudaMemcpyDeviceToDevice));
        }
    }
}

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
