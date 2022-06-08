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

#include <morpheus/messages/multi_response_probs.hpp>

#include <pysrf/node.hpp>
#include <srf/segment/builder.hpp>

#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** FilterDetectionStage********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class FilterDetectionsStage : public srf::pysrf::PythonNode<std::shared_ptr<MultiResponseProbsMessage>,
                                                            std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t =
        srf::pysrf::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    FilterDetectionsStage(float threshold);

  private:
    subscribe_fn_t build_operator();

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
    static std::shared_ptr<srf::segment::Object<FilterDetectionsStage>> init(srf::segment::Builder &parent,
                                                                             const std::string &name,
                                                                             float threshold);
};

#pragma GCC visibility pop
}  // namespace morpheus
