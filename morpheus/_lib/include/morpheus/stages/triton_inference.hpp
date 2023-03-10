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

#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/objects/triton_in_out.hpp"

#include <boost/fiber/future/future.hpp>
#include <http_client.h>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/types.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceClientStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief Perform inference with Triton Inference Server.
 * This class specifies which inference implementation category (Ex: NLP/FIL) is needed for inferencing.
 */
class InferenceClientStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseProbsMessage>>
{
  public:
    using base_t =
        mrc::pymrc::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseProbsMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Inference Client Stage object
     *
     * @param model_name : Name of the model specifies which model can handle the inference requests that are sent to
     * Triton inference
     * @param server_url : Triton server URL.
     * @param force_convert_inputs : Instructs the stage to convert the incoming data to the same format that Triton is
     * expecting. If set to False, data will only be converted if it would not result in the loss of data.
     * @param use_shared_memory : Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using
     * CUDA IPC reduces network transfer time but requires that Morpheus and Triton are located on the same machine.
     * @param needs_logits : Determines if logits are required.
     * @param inout_mapping : Dictionary used to map pipeline input/output names to Triton input/output names. Use this
     * if the Morpheus names do not match the model.
     */
    InferenceClientStage(std::string model_name,
                         std::string server_url,
                         bool force_convert_inputs,
                         bool use_shared_memory,
                         bool needs_logits,
                         std::map<std::string, std::string> inout_mapping = {});

  private:
    /**
     * TODO(Documentation)
     */
    bool is_default_grpc_port(std::string& server_url);

    /**
     * TODO(Documentation)
     */
    void connect_with_server();

    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

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
struct InferenceClientStageInterfaceProxy
{
    /**
     * @brief Create and initialize a InferenceClientStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param model_name : Name of the model specifies which model can handle the inference requests that are sent to
     * Triton inference
     * @param server_url : Triton server URL.
     * @param force_convert_inputs : Instructs the stage to convert the incoming data to the same format that Triton is
     * expecting. If set to False, data will only be converted if it would not result in the loss of data.
     * @param use_shared_memory : Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using
     * CUDA IPC reduces network transfer time but requires that Morpheus and Triton are located on the same machine.
     * @param needs_logits : Determines if logits are required.
     * @param inout_mapping : Dictionary used to map pipeline input/output names to Triton input/output names. Use this
     * if the Morpheus names do not match the model.
     * @return std::shared_ptr<mrc::segment::Object<InferenceClientStage>>
     */
    static std::shared_ptr<mrc::segment::Object<InferenceClientStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::string model_name,
        std::string server_url,
        bool force_convert_inputs,
        bool use_shared_memory,
        bool needs_logits,
        std::map<std::string, std::string> inout_mapping);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
