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

#include "morpheus/stages/preprocess_fil.hpp"

#include "mrc/segment/object.hpp"  // for Object

#include "morpheus/messages/control.hpp"                      // for ControlMessage
#include "morpheus/messages/memory/inference_memory_fil.hpp"  // for InferenceMemoryFIL
#include "morpheus/messages/memory/tensor_memory.hpp"         // for TensorMemory
#include "morpheus/messages/meta.hpp"                         // for MessageMeta
#include "morpheus/messages/multi.hpp"                        // for MultiMessage
#include "morpheus/messages/multi_inference.hpp"              // for MultiInferenceMessage
#include "morpheus/objects/dev_mem_info.hpp"                  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"                         // for DType, TypeId
#include "morpheus/objects/table_info.hpp"                    // for TableInfo, MutableTableInfo
#include "morpheus/objects/tensor.hpp"                        // for Tensor
#include "morpheus/objects/tensor_object.hpp"                 // for TensorObject
#include "morpheus/types.hpp"                                 // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"                   // for MatxUtil

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column.hpp>       // for column
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/types.hpp>               // for type_id, data_type
#include <cudf/unary.hpp>               // for cast
#include <mrc/cuda/common.hpp>          // for __check_cuda_errors, MRC_CHECK_CUDA
#include <mrc/segment/builder.hpp>      // for Builder
#include <pybind11/gil.h>               // for gil_scoped_acquire
#include <pybind11/pybind11.h>          // for object_api::operator(), operator""_a, arg
#include <pybind11/pytypes.h>           // for object, str, object_api, generic_item, literals
#include <rmm/cuda_stream_view.hpp>     // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>        // for device_buffer

#include <algorithm>    // for find
#include <cstddef>      // for size_t
#include <memory>       // for shared_ptr, __shared_ptr_access, allocator, mak...
#include <type_traits>  // for is_same_v
#include <utility>      // for move

namespace morpheus {
// Component public implementations
// ************ PreprocessFILStage ************************* //
template <typename InputT, typename OutputT>
PreprocessFILStage<InputT, OutputT>::PreprocessFILStage(const std::vector<std::string>& features) :
  base_t(rxcpp::operators::map([this](sink_type_t x) {
      return this->on_data(std::move(x));
  })),
  m_fea_cols(std::move(features))
{}

template <typename InputT, typename OutputT>
void PreprocessFILStage<InputT, OutputT>::transform_bad_columns(std::vector<std::string>& fea_cols,
                                                                morpheus::MutableTableInfo& mutable_info)
{
    auto df_meta_col_names = mutable_info.get_column_names();
    std::vector<std::string> bad_cols;
    // Only check the feature columns. Leave the rest unchanged
    for (auto& fea_col : fea_cols)
    {
        // Find the index of the column in the dataframe
        auto col_idx =
            std::find(df_meta_col_names.begin(), df_meta_col_names.end(), fea_col) - df_meta_col_names.begin();

        if (col_idx == df_meta_col_names.size())
        {
            // This feature was not found. Ignore it.
            continue;
        }

        if (mutable_info.get_column(col_idx).type().id() == cudf::type_id::STRING)
        {
            bad_cols.push_back(fea_col);
        }
    }

    // Exit early if there is nothing to do
    if (!bad_cols.empty())
    {
        // Need to ensure all string columns have been converted to numbers. This requires running a
        // regex which is too difficult to do from C++ at this time. So grab the GIL, make the
        // conversions, and release. This is horribly inefficient, but so is the JSON lines format for
        // this workflow
        using namespace pybind11::literals;
        pybind11::gil_scoped_acquire gil;

        // pybind11::object df = x->meta->get_py_table();
        auto pdf = mutable_info.checkout_obj();
        auto& df = *pdf;

        std::string regex = R"((\d+))";

        for (auto c : bad_cols)
        {
            df[pybind11::str(c)] = df[pybind11::str(c)]
                                       .attr("str")
                                       .attr("extract")(pybind11::str(regex), "expand"_a = true)
                                       .attr("astype")(pybind11::str("float32"));
        }

        mutable_info.return_obj(std::move(pdf));
    }
}

template <typename InputT, typename OutputT>
TableInfo PreprocessFILStage<InputT, OutputT>::fix_bad_columns(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
    {
        {
            // Get the mutable info for the entire meta object so we only do this once per dataframe
            auto mutable_info = x->meta->get_mutable_info();
            transform_bad_columns(this->m_fea_cols, mutable_info);
        }

        // Now re-get the meta
        return x->get_meta(m_fea_cols);
    }
    else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
    {
        {
            // Get the mutable info for the entire meta object so we only do this once per dataframe
            auto mutable_info = x->payload()->get_mutable_info();
            transform_bad_columns(this->m_fea_cols, mutable_info);
        }

        // Now re-get the meta
        return x->payload()->get_info(m_fea_cols);
    }
    else
    {
        // sink_type_t not supported
        static_assert(!sizeof(sink_type_t), "PreProcessFILStage receives unsupported input type");
    }
}

template <typename InputT, typename OutputT>
PreprocessFILStage<InputT, OutputT>::source_type_t PreprocessFILStage<InputT, OutputT>::on_data(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<MultiMessage>>)
    {
        return on_multi_message(x);
    }
    else if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<ControlMessage>>)
    {
        return on_control_message(x);
    }
    else
    {
        // sink_type_t not supported
        static_assert(!sizeof(sink_type_t), "PreProcessFILStage receives unsupported input type");
    }
}

template <>
std::shared_ptr<MultiInferenceMessage> PreprocessFILStage<MultiMessage, MultiInferenceMessage>::on_multi_message(
    std::shared_ptr<MultiMessage> x)
{
    auto packed_data = std::make_shared<rmm::device_buffer>(m_fea_cols.size() * x->mess_count * sizeof(float),
                                                            rmm::cuda_stream_per_thread);
    auto df_meta     = this->fix_bad_columns(x);
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
    auto transposed_data = MatxUtil::transpose(DevMemInfo{packed_data,
                                                          TypeId::FLOAT32,
                                                          {static_cast<TensorIndex>(m_fea_cols.size()), x->mess_count},
                                                          {x->mess_count, 1}});

    // Create the tensor which will be row-major and size [row_count, fea_len]
    auto input__0 = Tensor::create(
        transposed_data, DType::create<float>(), {x->mess_count, static_cast<TensorIndex>(m_fea_cols.size())}, {}, 0);

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

template <>
std::shared_ptr<ControlMessage> PreprocessFILStage<ControlMessage, ControlMessage>::on_control_message(
    std::shared_ptr<ControlMessage> x)

{
    auto df_meta        = this->fix_bad_columns(x);
    const auto num_rows = df_meta.num_rows();

    auto packed_data =
        std::make_shared<rmm::device_buffer>(m_fea_cols.size() * num_rows * sizeof(float), rmm::cuda_stream_per_thread);

    for (size_t i = 0; i < df_meta.num_columns(); ++i)
    {
        auto curr_col = df_meta.get_column(i);
        auto curr_ptr = static_cast<float*>(packed_data->data()) + i * num_rows;

        // Check if we are something other than float
        if (curr_col.type().id() != cudf::type_id::FLOAT32)
        {
            auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

            // Do the copy here before it goes out of scope
            MRC_CHECK_CUDA(
                cudaMemcpy(curr_ptr, float_data.data->data(), num_rows * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        else
        {
            MRC_CHECK_CUDA(cudaMemcpy(
                curr_ptr, curr_col.template data<float>(), num_rows * sizeof(float), cudaMemcpyDeviceToDevice));
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
    auto seq_ids      = Tensor::create(
        MatxUtil::create_seq_ids(num_rows, m_fea_cols.size(), seq_id_dtype.type_id(), input__0.get_memory(), 0),
        seq_id_dtype,
        {num_rows, 3},
        {},
        0);

    // Build the results
    auto memory = std::make_shared<TensorMemory>(num_rows);
    memory->set_tensor("input__0", std::move(input__0));
    memory->set_tensor("seq_ids", std::move(seq_ids));
    x->tensors(memory);

    return x;
}

template class PreprocessFILStage<MultiMessage, MultiInferenceMessage>;
template class PreprocessFILStage<ControlMessage, ControlMessage>;

// ************ PreprocessFILStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessFILStageMM>> PreprocessFILStageInterfaceProxy::init_multi(
    mrc::segment::Builder& builder, const std::string& name, const std::vector<std::string>& features)
{
    auto stage = builder.construct_object<PreprocessFILStageMM>(name, features);

    return stage;
}

std::shared_ptr<mrc::segment::Object<PreprocessFILStageCM>> PreprocessFILStageInterfaceProxy::init_cm(
    mrc::segment::Builder& builder, const std::string& name, const std::vector<std::string>& features)
{
    auto stage = builder.construct_object<PreprocessFILStageCM>(name, features);

    return stage;
}
}  // namespace morpheus
