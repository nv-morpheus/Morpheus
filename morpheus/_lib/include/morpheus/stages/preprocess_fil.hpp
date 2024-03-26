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

#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include <cudf/column/column.hpp>       // for column, column::contents
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <boost/fiber/context.hpp>
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
class PreprocessFILStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>
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
                // Make sure to
                auto df_meta = this->fix_bad_columns(x);

                auto packed_data = std::make_shared<rmm::device_buffer>(
                    m_fea_cols.size() * x->mess_count * sizeof(float), rmm::cuda_stream_per_thread);

                for (size_t i = 0; i < df_meta.num_columns(); ++i)
                {
                    auto curr_col = df_meta.get_column(i);

                    auto curr_ptr = static_cast<float*>(packed_data->data()) + i * df_meta.num_rows();

                    // Check if we are something other than float
                    if (curr_col.type().id() != cudf::type_id::FLOAT32)
                    {
                        auto float_data = cudf::cast(curr_col, cudf::data_type(cudf::type_id::FLOAT32))->release();

                        // Do the copy here before it goes out of scope
                        MRC_CHECK_CUDA(cudaMemcpy(curr_ptr,
                                                  float_data.data->data(),
                                                  df_meta.num_rows() * sizeof(float),
                                                  cudaMemcpyDeviceToDevice));
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
                auto seq_ids      = Tensor::create(MatxUtil::create_seq_ids(x->mess_count,
                                                                       m_fea_cols.size(),
                                                                       seq_id_dtype.type_id(),
                                                                       input__0.get_memory(),
                                                                       x->mess_offset),
                                              seq_id_dtype,
                                                   {x->mess_count, 3},
                                                   {},
                                              0);

                // Build the results
                auto memory =
                    std::make_shared<InferenceMemoryFIL>(x->mess_count, std::move(input__0), std::move(seq_ids));

                auto next = std::make_shared<MultiInferenceMessage>(
                    x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                output.on_next(std::move(next));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

template <typename InputT, typename OutputT>
TableInfo PreprocessFILStage<InputT, OutputT>::fix_bad_columns(sink_type_t x)
{
    std::vector<std::string> bad_cols;

    {
        // Get the mutable info for the entire meta object so we only do this once per dataframe
        auto mutable_info      = x->meta->get_mutable_info();
        auto df_meta_col_names = mutable_info.get_column_names();

        // Only check the feature columns. Leave the rest unchanged
        for (auto& fea_col : m_fea_cols)
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

    // Now re-get the meta
    return x->get_meta(m_fea_cols);
}

/****** PreprocessFILStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreprocessFILStageInterfaceProxy
{
    /**
     * @brief Create and initialize a PreprocessFILStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param features : Reference to the features that are required for model inference
     * @return std::shared_ptr<mrc::segment::Object<PreprocessFILStage>>
     */
    static std::shared_ptr<mrc::segment::Object<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>> init(mrc::segment::Builder& builder,
                                                                          const std::string& name,
                                                                          const std::vector<std::string>& features);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
