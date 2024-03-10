/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/messages/multi_response.hpp"  // for MultiResponseMessage
#include "morpheus/objects/triton_in_out.hpp"
#include "morpheus/types.hpp"

#include <boost/fiber/context.hpp>
#include <boost/fiber/future/future.hpp>
#include <http_client.h>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>
#include <pymrc/asyncio_runnable.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Checks the status object returned by a Triton client call logging any potential errors.
 *
 */
#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

// Component-private free functions.
void InferenceClientStage__check_triton_errors(triton::client::Error status,
                                               const std::string& methodName,
                                               const std::string& filename,
                                               const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg = MORPHEUS_CONCAT_STR("Triton Error while executing '"
                                                  << methodName << "'. Error: " + status.Message() << "\n"
                                                  << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceClientStage********************************/

class ITritonClient
{
  public:
    virtual triton::client::Error is_server_live(bool* live)                                           = 0;
    virtual triton::client::Error is_server_ready(bool* ready)                                         = 0;
    virtual triton::client::Error is_model_ready(bool* ready, std::string& model_name)                 = 0;
    virtual triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) = 0;
    virtual triton::client::Error model_config(std::string* model_config, std::string& model_name)     = 0;
    virtual triton::client::Error async_infer(
        triton::client::InferenceServerHttpClient::OnCompleteFn callback,
        const triton::client::InferOptions& options,
        const std::vector<triton::client::InferInput*>& inputs,
        const std::vector<const triton::client::InferRequestedOutput*>& outputs) = 0;
};

class HttpTritonClient : public ITritonClient
{
  private:
    static bool is_default_grpc_port(std::string& server_url)
    {
        // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
        size_t colon_loc = server_url.find_last_of(':');

        if (colon_loc == -1)
        {
            return false;
        }

        // Check if the port matches 8001
        if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001")
        {
            return false;
        }

        // It matches, change to 8000
        server_url = server_url.substr(0, colon_loc) + ":8000";

        return true;
    }

  public:
    HttpTritonClient(std::string server_url)
    {
        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, server_url, false));

        bool is_server_live;

        auto status = client->IsServerLive(&is_server_live);

        if (not status.IsOk())
        {
            std::string new_server_url = server_url;
            if (is_default_grpc_port(new_server_url))
            {
                LOG(WARNING) << "Failed to connect to Triton at '" << server_url
                             << "'. Default gRPC port of (8001) was detected but C++ "
                                "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

                // We are using the default gRPC port, try the default HTTP
                std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

                CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&unique_client, new_server_url, false));

                client = std::move(unique_client);

                status = client->IsServerLive(&is_server_live);
            }
            else if (status.Message().find("Unsupported protocol") != std::string::npos)
            {
                throw std::runtime_error(MORPHEUS_CONCAT_STR(
                    "Failed to connect to Triton at '"
                    << server_url
                    << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                       "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                       "specified the HTTP port (Default 8000)."));
            }

            if (not status.IsOk())
                throw std::runtime_error(MORPHEUS_CONCAT_STR(
                    "Unable to connect to Triton at '"
                    << server_url << "'. Check the URL and port and ensure the server is running."));
        }

        m_client = std::move(client);
    }

    triton::client::Error is_server_live(bool* live) override
    {
        return m_client->IsServerLive(live);
    }

    triton::client::Error is_server_ready(bool* ready) override
    {
        return m_client->IsServerReady(ready);
    }

    triton::client::Error is_model_ready(bool* ready, std::string& model_name) override
    {
        return m_client->IsModelReady(ready, model_name);
    }

    triton::client::Error model_config(std::string* model_config, std::string& model_name) override
    {
        return m_client->ModelConfig(model_config, model_name);
    }

    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override
    {
        return m_client->ModelMetadata(model_metadata, model_name);
    }

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<triton::client::InferInput*>& inputs,
                                      const std::vector<const triton::client::InferRequestedOutput*>& outputs) override
    {
        return m_client->AsyncInfer(callback, options, inputs, outputs);
    }

  private:
    std::unique_ptr<triton::client::InferenceServerHttpClient> m_client;
};

struct TritonInferenceClient
{
  private:
    std::string m_server_url;
    std::string m_model_name;
    TensorIndex m_max_batch_size = -1;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    std::shared_ptr<ITritonClient> m_client;

  public:
    TritonInferenceClient(std::shared_ptr<ITritonClient> client, std::string model_name);

    std::map<std::string, std::string> get_input_mappings(std::map<std::string, std::string> input_map_overrides);

    std::map<std::string, std::string> get_output_mappings(std::map<std::string, std::string> output_map_overrides);

    mrc::coroutines::Task<TensorMap> infer(TensorMap&& inputs);
};

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
  : public mrc::pymrc::AsyncioRunnable<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using sink_type_t   = std::shared_ptr<MultiInferenceMessage>;
    using source_type_t = std::shared_ptr<MultiResponseMessage>;

    /**
     * @brief Construct a new Inference Client Stage object
     *
     * @param create_client : Create's a Triton client instance.
     * @param model_name : Name of the model specifies which model can handle the inference requests that are sent to
     * Triton inference
     * @param needs_logits : Determines if logits are required.
     * @param inout_mapping : Dictionary used to map pipeline input/output names to Triton input/output names. Use this
     * if the Morpheus names do not match the model.
     */
    InferenceClientStage(std::function<std::shared_ptr<ITritonClient>()> create_client,
                         std::string model_name,
                         bool needs_logits,
                         std::map<std::string, std::string> input_mapping  = {},
                         std::map<std::string, std::string> output_mapping = {});

    /**
     * TODO(Documentation)
     */
    static bool is_default_grpc_port(std::string& server_url);

    /**
     * TODO(Documentation)
     */
    mrc::coroutines::AsyncGenerator<std::shared_ptr<MultiResponseMessage>> on_data(
        std::shared_ptr<MultiInferenceMessage>&& data, std::shared_ptr<mrc::coroutines::Scheduler> on) override;

  private:
    std::shared_ptr<TritonInferenceClient> get_client();
    void reset_client();

    std::string m_model_name;
    std::function<std::shared_ptr<ITritonClient>()> m_create_client;
    bool m_needs_logits{true};
    std::map<std::string, std::string> m_input_mapping;
    std::map<std::string, std::string> m_output_mapping;

    // // Below are settings created during handshake with server
    // // std::shared_ptr<triton::client::InferenceServerHttpClient> m_client;
    // std::vector<TritonInOut> m_model_inputs;
    // std::vector<TritonInOut> m_model_outputs;
    // triton::client::InferOptions m_options;
    // TensorIndex m_max_batch_size{-1};
    std::mutex m_client_mutex;
    std::shared_ptr<TritonInferenceClient> m_client;

    int32_t m_retry_max = 10;
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
        bool needs_logits,
        std::map<std::string, std::string> input_mapping,
        std::map<std::string, std::string> output_mapping);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
