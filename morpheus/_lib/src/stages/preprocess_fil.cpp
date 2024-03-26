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

#include "mrc/segment/object.hpp"

#include "morpheus/messages/control.hpp"
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
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor, arg
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <algorithm>  // for std::find
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

namespace morpheus {
// Component public implementations
// ************ PreprocessFILStage ************************* //
void transform_bad_columns(std::vector<std::string>& fea_cols,
                           morpheus::MutableTableInfo& mutable_info,
                           std::vector<std::string>& df_meta_col_names)
{
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
// ************ PreprocessFILStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>>
PreprocessFILStageInterfaceProxy::init_multi(mrc::segment::Builder& builder,
                                             const std::string& name,
                                             const std::vector<std::string>& features)
{
    auto stage = builder.construct_object<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>(name, features);

    return stage;
}

std::shared_ptr<mrc::segment::Object<PreprocessFILStage<ControlMessage, ControlMessage>>>
PreprocessFILStageInterfaceProxy::init_cm(mrc::segment::Builder& builder,
                                          const std::string& name,
                                          const std::vector<std::string>& features)
{
    auto stage = builder.construct_object<PreprocessFILStage<ControlMessage, ControlMessage>>(name, features);

    return stage;
}
}  // namespace morpheus
