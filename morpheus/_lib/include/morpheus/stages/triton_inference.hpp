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

#include <morpheus/messages/multi_inference.hpp>
#include <morpheus/messages/multi_response.hpp>
#include <morpheus/objects/triton_in_out.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/node.hpp>

#include <http_client.h>

#include <string>
#include <memory>


namespace morpheus {
    /****** Component public implementations *******************/
    /****** InferenceClientStage********************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class InferenceClientStage
            : public neo::pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>> {
    public:
        using base_t = neo::pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>;
        using base_t::operator_fn_t;
        using base_t::reader_type_t;
        using base_t::writer_type_t;

        InferenceClientStage(const neo::Segment &parent,
                             const std::string &name,
                             std::string model_name,
                             std::string server_url,
                             bool force_convert_inputs,
                             bool use_shared_memory,
                             bool needs_logits,
                             std::map<std::string, std::string> inout_mapping = {});

    private:
        /**
         * TODO(Documentation)
         */
        bool is_default_grpc_port(std::string &server_url);

        /**
         * TODO(Documentation)
         */
        void connect_with_server();

        /**
         * TODO(Documentation)
         */
        operator_fn_t build_operator();

        std::string m_model_name;
        std::string m_server_url;
        bool m_force_convert_inputs;
        bool m_use_shared_memory;
        bool m_needs_logits{true};
        std::map<std::string, std::string> m_inout_mapping;

        // Below are settings created during handshake with server
        // std::shared_ptr<triton::client::InferenceServerHttpClient> m_client;
        std::vector<TritonInOut> m_model_inputs;
        std::vector<TritonInOut> m_model_outputs;
        triton::client::InferOptions m_options;
        int m_max_batch_size{-1};
    };


    /****** InferenceClientStageInferenceProxy******************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct InferenceClientStageInterfaceProxy {
        /**
         * @brief Create and initialize a InferenceClientStage, and return the result.
         */
        static std::shared_ptr<InferenceClientStage> init(neo::Segment &parent,
                                                          const std::string &name,
                                                          std::string model_name,
                                                          std::string server_url,
                                                          bool force_convert_inputs,
                                                          bool use_shared_memory,
                                                          bool needs_logits,
                                                          std::map<std::string, std::string> inout_mapping);
    };
#pragma GCC visibility pop
}