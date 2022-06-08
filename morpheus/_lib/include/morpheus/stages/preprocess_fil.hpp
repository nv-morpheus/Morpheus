/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/messages/multi.hpp>
#include <morpheus/messages/multi_inference.hpp>

#include <pysrf/node.hpp>
#include <srf/segment/builder.hpp>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** PreprocessFILStage**********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class PreprocessFILStage
  : public srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreprocessFILStage(const std::vector<std::string>& features);

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::vector<std::string> m_fea_cols;
    std::string m_vocab_file;
};

/****** PreprocessFILStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreprocessFILStageInterfaceProxy
{
    /**
     * @brief Create and initialize a PreprocessFILStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<PreprocessFILStage>> init(srf::segment::Builder& parent,
                                                                          const std::string& name,
                                                                          const std::vector<std::string>& features);
};
#pragma GCC visibility pop
}  // namespace morpheus
