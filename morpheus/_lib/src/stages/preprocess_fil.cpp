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

#include "morpheus/stages/preprocess_fil.hpp"

#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/meta.hpp"         // for MessageMeta
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/types.hpp"                  // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpyDeviceToDevice
#include <cudf/column/column.hpp>       // for column, column::contents
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <mrc/segment/builder.hpp>
#include <pybind11/cast.h>  // for object_api::operator(), operator""_a
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor, arg
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <array>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <utility>

namespace morpheus {
// Component public implementations
// ************ PreprocessFILStage ************************* //
PreprocessFILStage::PreprocessFILStage(const std::vector<std::string>& features) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_fea_cols(std::move(features))
{}

PreprocessFILStage::subscribe_fn_t PreprocessFILStage::build_operator()
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
                                                  curr_col.data<float>(),
                                                  df_meta.num_rows() * sizeof(float),
                                                  cudaMemcpyDeviceToDevice));
                    }
                }

                // Need to do a transpose here
                auto transposed_data =
                    MatxUtil::transpose(DevMemInfo{packed_data,
                                                   TypeId::FLOAT32,
                                                   {x->mess_count, static_cast<TensorIndex>(m_fea_cols.size())},
                                                   {1, x->mess_count}});

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
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

TableInfo PreprocessFILStage::fix_bad_columns(sink_type_t x)
{
    std::vector<std::string> bad_cols;

    {
        // TODO(MDD): Add some sort of lock here to prevent fixing columns after they have been accessed
        auto df_meta           = x->get_meta(m_fea_cols);
        auto df_meta_col_names = df_meta.get_column_names();

        for (size_t i = 0; i < df_meta.num_columns(); ++i)
        {
            if (df_meta.get_column(i).type().id() == cudf::type_id::STRING)
            {
                bad_cols.push_back(df_meta_col_names[i]);
            }
        }

        // Exit early if there is nothing to do
        if (bad_cols.empty())
        {
            return df_meta;
        }
    }

    // Need to ensure all string columns have been converted to numbers. This requires running a
    // regex which is too difficult to do from C++ at this time. So grab the GIL, make the
    // conversions, and release. This is horribly inefficient, but so is the JSON lines format for
    // this workflow
    {
        // Get the mutable info for the entire meta object so we only do this once per dataframe
        auto mutable_info = x->meta->get_mutable_info();

        using namespace pybind11::literals;
        pybind11::gil_scoped_acquire gil;

        // pybind11::object df = x->meta->get_py_table();
        auto df = mutable_info.checkout_obj();

        std::string regex = R"((\d+))";

        for (auto c : bad_cols)
        {
            df[pybind11::str(c)] = df[pybind11::str(c)]
                                       .attr("str")
                                       .attr("extract")(pybind11::str(regex), "expand"_a = true)
                                       .attr("astype")(pybind11::str("float32"));
        }

        mutable_info.return_obj(std::move(df));
    }

    // Now re-get the meta
    return x->get_meta(m_fea_cols);
}

// ************ PreprocessFILStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessFILStage>> PreprocessFILStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, const std::vector<std::string>& features)
{
    auto stage = builder.construct_object<PreprocessFILStage>(name, features);

    return stage;
}
}  // namespace morpheus
