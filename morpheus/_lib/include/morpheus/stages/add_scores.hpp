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

#include <string>
#include <memory>


namespace morpheus {
    /****** Component public implementations *******************/
    /****** AddScoresStage********************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class AddScoresStage : public neo::pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>,
            std::shared_ptr<MultiResponseProbsMessage>> {
    public:
        using base_t =
        neo::pyneo::PythonNode<std::shared_ptr<MultiResponseProbsMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
        using base_t::operator_fn_t;
        using base_t::reader_type_t;
        using base_t::writer_type_t;

        AddScoresStage(const neo::Segment &parent,
                       const std::string &name,
                       std::size_t num_class_labels,
                       std::map<std::size_t, std::string> idx2label);

        /**
         * TODO(Documentation)
         */
        operator_fn_t build_operator();

        std::size_t m_num_class_labels;
        std::map<std::size_t, std::string> m_idx2label;
    };

    /****** AddScoresStageInterfaceProxy******************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct AddScoresStageInterfaceProxy {

        /**
         * @brief Create and initialize a AddScoresStage, and return the result.
         */
        static std::shared_ptr<AddScoresStage> init(neo::Segment &parent,
                                                    const std::string &name,
                                                    std::size_t num_class_labels,
                                                    std::map<std::size_t, std::string> idx2label);
    };

#pragma GCC visibility pop
}