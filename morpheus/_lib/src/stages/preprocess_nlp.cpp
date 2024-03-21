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

#include "morpheus/stages/preprocess_nlp.hpp"

#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/inference_memory.hpp"  // for InferenceMemory
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo
#include "morpheus/objects/tensor.hpp"
#include "morpheus/types.hpp"  // for TensorIndex, TensorMap
#include "morpheus/utilities/matx_util.hpp"

#include <cudf/column/column.hpp>  // for column, column::contents
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <mrc/segment/builder.hpp>
#include <nvtext/normalize.hpp>
#include <nvtext/subword_tokenize.hpp>
#include <pymrc/node.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>  // for device_buffer

#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>

namespace morpheus {
// Component public implementations
// ************ PreprocessNLPStage ************************* //

// ************ PreprocessNLPStageInterfaceProxy *********** //
std::shared_ptr<mrc::segment::Object<PreprocessNLPStage<MultiMessage, MultiInferenceMessage>>> PreprocessNLPStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string vocab_hash_file,
    uint32_t sequence_length,
    bool truncation,
    bool do_lower_case,
    bool add_special_token,
    int stride,
    std::string column)
{
    auto stage = builder.construct_object<PreprocessNLPStage<MultiMessage, MultiInferenceMessage>>(
        name, vocab_hash_file, sequence_length, truncation, do_lower_case, add_special_token, stride, column);

    return stage;
}
}  // namespace morpheus
