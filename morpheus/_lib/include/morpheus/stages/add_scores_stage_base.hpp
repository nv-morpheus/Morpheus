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
#include <rxcpp/rx.hpp>

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** AddClassificationStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief Base class for both `AddScoresStage` and `AddClassificationStage`
 */
class AddScoresStageBase
  : public mrc::pymrc::PythonNode<std::shared_ptr<MultiResponseMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MultiResponseMessage>, std::shared_ptr<MultiResponseMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Add Classifications Stage object
     *
     * @param threshold : Threshold to consider true/false for each class
     * @param idx2label : Index to classification labels map
     */
    AddScoresStageBase(std::map<std::size_t, std::string> idx2label, std::optional<float> threshold);

  private:
    /**
     * Called every time a message is passed to this stage
     */
    source_type_t on_data(sink_type_t x);

    std::map<std::size_t, std::string> m_idx2label;
    std::optional<float> m_threshold;

    // The minimum number of columns needed to extract the label data
    std::size_t m_min_col_count;
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
