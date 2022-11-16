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

#include "morpheus/messages/multi_response_probs.hpp"

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** AddScoresStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
*/

/**
 * Add probability scores to each message.
 * 
 * Add score labels based on probabilities calculated in inference stage. Label indexes will be looked up in
 * the idx2label property.
 */
#pragma GCC visibility push(default)
class AddScoresStage : public srf::pysrf::PythonNode<std::shared_ptr<MultiResponseProbsMessage>,
                                                     std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t =
        srf::pysrf::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * Constructor of a class `AddScoresStage`.
     * 
     * @param num_class_labels : Number of classification labels.
     * @param idx2label : Index to classification labels map.
    */
    AddScoresStage(std::size_t num_class_labels, std::map<std::size_t, std::string> idx2label);

    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::size_t m_num_class_labels;
    std::map<std::size_t, std::string> m_idx2label;
};

/****** AddScoresStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct AddScoresStageInterfaceProxy
{
    /**
     * @brief Create and initialize a AddScoresStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<AddScoresStage>> init(srf::segment::Builder &builder,
                                                                      const std::string &name,
                                                                      std::size_t num_class_labels,
                                                                      std::map<std::size_t, std::string> idx2label);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
