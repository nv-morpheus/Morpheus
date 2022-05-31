/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pyneo/node.hpp>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** PreprocessNLPStage**********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class PreprocessNLPStage
  : public neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessNLPStage(const neo::Segment &parent,
                       const std::string &name,
                       std::string vocab_hash_file,
                       uint32_t sequence_length,
                       bool truncation,
                       bool do_lower_case,
                       bool add_special_token,
                       int stride = -1);

  private:
    /**
     * TODO(Documentation)
     */
    operator_fn_t build_operator();

    std::string m_vocab_hash_file;
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
    static std::shared_ptr<PreprocessNLPStage> init(neo::Segment &parent,
                                                    const std::string &name,
                                                    std::string vocab_hash_file,
                                                    uint32_t sequence_length,
                                                    bool truncation,
                                                    bool do_lower_case,
                                                    bool add_special_token,
                                                    int stride = -1);
};
#pragma GCC visibility pop
}  // namespace morpheus
