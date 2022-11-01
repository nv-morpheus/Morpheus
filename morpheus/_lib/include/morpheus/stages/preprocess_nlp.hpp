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

#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <cstdint>  // for uint32_t
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** PreprocessNLPStage**********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class PreprocessNLPStage
  : public srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreprocessNLPStage(std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       std::string column,
                       int stride = -1);

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::string m_vocab_hash_file;
    std::string m_column;
    uint32_t m_sequence_length;
    bool m_truncation;
    bool m_do_lower_case;
    bool m_add_special_token;
    int m_stride{-1};
};

/****** PreprocessNLPStageInferenceProxy********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreprocessNLPStageInterfaceProxy
{
    /**
     * @brief Create and initialize a ProcessNLPStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<PreprocessNLPStage>> init(srf::segment::Builder &builder,
                                                                          const std::string &name,
                                                                          std::string vocab_hash_file,
                                                                          uint32_t sequence_length,
                                                                          bool truncation,
                                                                          bool do_lower_case,
                                                                          bool add_special_token,
                                                                          const std::string column,
                                                                          int stride = -1);
};
#pragma GCC visibility pop
}  // namespace morpheus
