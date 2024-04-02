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

    std::shared_ptr<OutputT> on_data(std::shared_ptr<InputT> x);

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::shared_ptr<MultiInferenceMessage> on_multi_message(std::shared_ptr<MultiMessage> x);
    std::shared_ptr<ControlMessage> on_control_message(std::shared_ptr<ControlMessage> x);

    TableInfo fix_bad_columns(sink_type_t x);

    std::vector<std::string> m_fea_cols;
    std::string m_vocab_file;
};

using PreprocessFILStageMM = // NOLINT(readability-identifier-naming)
    PreprocessFILStage<MultiMessage, MultiInferenceMessage>;
using PreprocessFILStageCC = // NOLINT(readability-identifier-naming)
    PreprocessFILStage<ControlMessage, ControlMessage>;  

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
