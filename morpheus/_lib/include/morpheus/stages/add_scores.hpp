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

#include "morpheus/stages/add_scores_stage_base.hpp"

#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <string>

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
class AddScoresStage : public AddScoresStageBase
{
  public:
    /**
     * @brief Construct a new Add Scores Stage object
     *
     * @param idx2label : Index to classification labels map
     */
    AddScoresStage(std::map<std::size_t, std::string> idx2label);
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
