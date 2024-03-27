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
#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/utilities/matx_util.hpp"

#include <boost/fiber/context.hpp>
#include <cudf/column/column.hpp>       // for column, column::contents
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {

/****** Component public implementations *******************/
/****** PreprocessFILStage**********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief FIL input data for inference
 */
template <typename InputT, typename OutputT>
class PreprocessFILStage : public mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Constructor for a class `PreprocessFILStage`
     *
     * @param features : Reference to the features that are required for model inference
     */
    PreprocessFILStage(const std::vector<std::string>& features);

    std::shared_ptr<OutputT> pre_process_batch(std::shared_ptr<InputT> x);

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    TableInfo fix_bad_columns(sink_type_t x);

    std::vector<std::string> m_fea_cols;
    std::string m_vocab_file;
};

template <typename InputT, typename OutputT>
PreprocessFILStage<InputT, OutputT>::PreprocessFILStage(const std::vector<std::string>& features) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_fea_cols(std::move(features))
{}

template <typename InputT, typename OutputT>
PreprocessFILStage<InputT, OutputT>::subscribe_fn_t PreprocessFILStage<InputT, OutputT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [&output, this](sink_type_t x) {
                auto next = this->pre_process_batch(x);
                output.on_next(std::move(next));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}  // namespace morpheus

void transform_bad_columns(std::vector<std::string>& fea_cols,
                           morpheus::MutableTableInfo& mutable_info,
                           std::vector<std::string>& df_meta_col_names);

template <typename InputT, typename OutputT>
TableInfo PreprocessFILStage<InputT, OutputT>::fix_bad_columns(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
    {
        {
            // Get the mutable info for the entire meta object so we only do this once per dataframe
            auto mutable_info      = x->meta->get_mutable_info();
            auto df_meta_col_names = mutable_info.get_column_names();

            transform_bad_columns(this->m_fea_cols, mutable_info, df_meta_col_names);
        }

        // Now re-get the meta
        return x->get_meta(m_fea_cols);
    }
    else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
    {
        {
            // Get the mutable info for the entire meta object so we only do this once per dataframe
            auto mutable_info      = x->payload()->get_mutable_info();
            auto df_meta_col_names = mutable_info.get_column_names();

            transform_bad_columns(this->m_fea_cols, mutable_info, df_meta_col_names);
        }
        
        // Now re-get the meta
        auto info        = x->payload()->get_info();
        auto num_columns = info.num_columns();
        return info.get_slice(0, num_columns, std::vector<std::string>(m_fea_cols));
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"PreProcessFILStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

template <typename InputT, typename OutputT>
std::shared_ptr<OutputT> PreprocessFILStage<InputT, OutputT>::pre_process_batch(std::shared_ptr<InputT> x)
{
    auto df_meta = this->fix_bad_columns(x);

    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
    {
        auto packed_data = std::make_shared<rmm::device_buffer>(m_fea_cols.size() * x->mess_count * sizeof(float),
                                                                rmm::cuda_stream_per_thread);

        for (size_t i = 0; i < df_meta.num_columns(); ++i)
        {
            auto curr_col = df_meta.get_column(i);

            auto curr_ptr = static_cast<float*>(packed_data->data()) + i * df_meta.num_rows();

            // Check if we are something other than float
            if (curr_col.type().id() != cudf::type_id::FLOAT32)
            {
                auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

                // Do the copy here before it goes out of scope
                MRC_CHECK_CUDA(cudaMemcpy(
                    curr_ptr, float_data.data->data(), df_meta.num_rows() * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            else
            {
                MRC_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                          curr_col.template data<float>(),
                                          df_meta.num_rows() * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
            }
        }

        // Need to convert from row major to column major
        // Easiest way to do this is to transpose the data from [fea_len, row_count] to [row_count, fea_len]
        auto transposed_data =
            MatxUtil::transpose(DevMemInfo{packed_data,
                                           TypeId::FLOAT32,
                                           {static_cast<TensorIndex>(m_fea_cols.size()), x->mess_count},
                                           {x->mess_count, 1}});

        // Create the tensor which will be row-major and size [row_count, fea_len]
        auto input__0 = Tensor::create(transposed_data,
                                       DType::create<float>(),
                                       {x->mess_count, static_cast<TensorIndex>(m_fea_cols.size())},
                                       {},
                                       0);

        auto seq_id_dtype = DType::create<TensorIndex>();
        auto seq_ids      = Tensor::create(
            MatxUtil::create_seq_ids(
                x->mess_count, m_fea_cols.size(), seq_id_dtype.type_id(), input__0.get_memory(), x->mess_offset),
            seq_id_dtype,
            {x->mess_count, 3},
            {},
            0);

        // Build the results
        auto memory = std::make_shared<InferenceMemoryFIL>(x->mess_count, std::move(input__0), std::move(seq_ids));

        auto next = std::make_shared<MultiInferenceMessage>(
            x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

        return next;
    }
    else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
    {
        auto num_rows    = x->payload()->get_info().num_rows();
        auto packed_data = std::make_shared<rmm::device_buffer>(m_fea_cols.size() * num_rows * sizeof(float),
                                                                rmm::cuda_stream_per_thread);

        for (size_t i = 0; i < df_meta.num_columns(); ++i)
        {
            auto curr_col = df_meta.get_column(i);

            auto curr_ptr = static_cast<float*>(packed_data->data()) + i * df_meta.num_rows();

            // Check if we are something other than float
            if (curr_col.type().id() != cudf::type_id::FLOAT32)
            {
                auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

                // Do the copy here before it goes out of scope
                MRC_CHECK_CUDA(cudaMemcpy(
                    curr_ptr, float_data.data->data(), df_meta.num_rows() * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            else
            {
                MRC_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                          curr_col.template data<float>(),
                                          df_meta.num_rows() * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
            }
        }

        // Need to convert from row major to column major
        // Easiest way to do this is to transpose the data from [fea_len, row_count] to [row_count, fea_len]
        auto transposed_data = MatxUtil::transpose(DevMemInfo{
            packed_data, TypeId::FLOAT32, {static_cast<TensorIndex>(m_fea_cols.size()), num_rows}, {num_rows, 1}});

        // Create the tensor which will be row-major and size [row_count, fea_len]
        auto input__0 = Tensor::create(
            transposed_data, DType::create<float>(), {num_rows, static_cast<TensorIndex>(m_fea_cols.size())}, {}, 0);

        auto seq_id_dtype = DType::create<TensorIndex>();
        auto seq_ids =
            Tensor::create(MatxUtil::create_seq_ids(
                               num_rows, m_fea_cols.size(), seq_id_dtype.type_id(), input__0.get_memory(), num_rows),
                           seq_id_dtype,
                           {num_rows, 3},
                           {},
                           0);

        // Build the results
        auto memory = std::make_shared<TensorMemory>(num_rows);
        memory->set_tensor("input__0", std::move(input__0));
        memory->set_tensor("seq_ids", std::move(seq_ids));
        auto next = x;
        next->tensors(memory);

        return next;
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"PreProcessFILStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

/****** PreprocessFILStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreprocessFILStageInterfaceProxy
{
    /**
     * @brief Create and initialize a PreprocessFILStage that receives MultiMessage and emits MultiInferenceMessage,
     * and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param features : Reference to the features that are required for model inference
     * @return std::shared_ptr<mrc::segment::Object<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>> init_multi(
        mrc::segment::Builder& builder, const std::string& name, const std::vector<std::string>& features);

    /**
     * @brief Create and initialize a PreprocessFILStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param features : Reference to the features that are required for model inference
     * @return std::shared_ptr<mrc::segment::Object<PreprocessFILStage<ControlMessage, ControlMessage>>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessFILStage<ControlMessage, ControlMessage>>> init_cm(
        mrc::segment::Builder& builder, const std::string& name, const std::vector<std::string>& features);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
