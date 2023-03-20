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

#pragma once

#include "morpheus/messages/multi_response.hpp"  // for MultiResponseMessage

#include <boost/fiber/future/future.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/types.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity

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

#pragma GCC visibility push(default)
/**
 * @brief Add probability scores to each message. Score labels based on probabilities calculated in inference stage.
 * Label indexes will be looked up in the idx2label property.
 */
class AddScoresStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MultiResponseMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MultiResponseMessage>, std::shared_ptr<MultiResponseMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Add Scores Stage object
     *
     * @param num_class_labels : Number of classification labels
     * @param idx2label : Index to classification labels map
     */
    AddScoresStage(std::map<std::size_t, std::string> idx2label);

    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    std::map<std::size_t, std::string> m_idx2label;

    std::size_t m_min_col_count;
};

/****** AddScoresStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct AddScoresStageInterfaceProxy
{
    /**
     * @brief Create and initialize a AddScoresStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param num_class_labels : Number of classification labels
     * @param idx2label : Index to classification labels map
     * @return std::shared_ptr<mrc::segment::Object<AddScoresStage>>
     */
    static std::shared_ptr<mrc::segment::Object<AddScoresStage>> init(mrc::segment::Builder& builder,
                                                                      const std::string& name,
                                                                      std::map<std::size_t, std::string> idx2label);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
