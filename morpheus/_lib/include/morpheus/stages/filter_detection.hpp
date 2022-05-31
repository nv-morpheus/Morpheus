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

#include <morpheus/messages/multi_response_probs.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** FilterDetectionStage********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class FilterDetectionsStage : public neo::pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>,
                                                            std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t =
        neo::pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    FilterDetectionsStage(const neo::Segment &parent, const std::string &name, float threshold);

  private:
    operator_fn_t build_operator();

    float m_threshold;
    std::size_t m_num_class_labels;
    std::map<std::size_t, std::string> m_idx2label;
};

/****** FilterDetectionStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct FilterDetectionStageInterfaceProxy
{
    /**
     * @brief Create and initialize a FilterDetectionStage, and return the result.
     */
    static std::shared_ptr<FilterDetectionsStage> init(neo::Segment &parent, const std::string &name, float threshold);
};

#pragma GCC visibility pop
}  // namespace morpheus
