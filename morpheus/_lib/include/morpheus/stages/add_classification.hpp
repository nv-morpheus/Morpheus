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

#include "morpheus/messages/multi_response_probs.hpp"

#include <mrc/channel/status.hpp>          // for Status
#include <mrc/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>  // for size_t
#include <map>
#include <memory>
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
 * @brief Add detected classifications to each message. Classification labels based on probabilities calculated in
 * inference stage. Label indexes will be looked up in the idx2label property.
 */
class AddClassificationsStage
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
     * @param num_class_labels : Number of classification labels
     * @param idx2label : Index to classification labels map
     */
    AddClassificationsStage(float threshold,
                            std::size_t num_class_labels,
                            std::map<std::size_t, std::string> idx2label,
                            std::string tensor_name = "probs");

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    float m_threshold;
    std::size_t m_num_class_labels;
    std::map<std::size_t, std::string> m_idx2label;
    std::string m_tensor_name;
};

/****** AddClassificationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */

struct AddClassificationStageInterfaceProxy
{
    /**
     * @brief Create and initialize a AddClassificationStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param threshold : Threshold to consider true/false for each class
     * @param num_class_labels : Number of classification labels
     * @param idx2label : Index to classification labels map
     * @return std::shared_ptr<mrc::segment::Object<AddClassificationsStage>>
     */
    static std::shared_ptr<mrc::segment::Object<AddClassificationsStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        float threshold,
        std::size_t num_class_labels,
        std::map<std::size_t, std::string> idx2label,
        std::string tensor_name);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
