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

#include "morpheus/export.h"
#include "morpheus/objects/triton_in_out.hpp"
#include "morpheus/stages/inference_client_stage.hpp"
#include "morpheus/types.hpp"

#include <http_client.h>
#include <mrc/coroutines/task.hpp>
#include <cstdint>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceClientStage********************************/

struct MORPHEUS_EXPORT TritonInferInput
{
    /**
     * @brief The name of the triton inference input
    */
    std::string name;

    /**
     * @brief The shape of the triton inference input
    */
    std::vector<int64_t> shape;

    /**
     * @brief The type of the triton inference input
    */
    std::string type;

    /**
     * @brief The triton inference input data
    */
    std::vector<uint8_t> data;
};

struct MORPHEUS_EXPORT TritonInferRequestedOutput
{
    std::string name;
};

class MORPHEUS_EXPORT ITritonClient
{
  public:

    /**
    * @brief Checks if Triton Server is live
    */
    virtual triton::client::Error is_server_live(bool* live) = 0;

    /**
    * @brief Checks if Triton Server is ready
    */
    virtual triton::client::Error is_server_ready(bool* ready) = 0;

    /**
    * @brief Checks if the given model is ready
    */
    virtual triton::client::Error is_model_ready(bool* ready, std::string& model_name) = 0;

    /**
    * @brief Gets metadata for the given model
    */
    virtual triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) = 0;

    /**
    * @brief Gets the config for the given model
    */
    virtual triton::client::Error model_config(std::string* model_config, std::string& model_name) = 0;

    /**
    * @brief Runs Triton Server inference given the model options, inputs, and outputs
    */
    virtual triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                              const triton::client::InferOptions& options,
                                              const std::vector<TritonInferInput>& inputs,
                                              const std::vector<TritonInferRequestedOutput>& outputs) = 0;
};

class MORPHEUS_EXPORT HttpTritonClient : public ITritonClient
{
  private:
    std::unique_ptr<triton::client::InferenceServerHttpClient> m_client;

  public:
    HttpTritonClient(std::string server_url);

    /**
    * @brief Checks if Triton Server is live using HTTP protocal
    */
    triton::client::Error is_server_live(bool* live) override;

    /**
    * @brief Checks if Triton Server is ready using HTTP protocal
    */
    triton::client::Error is_server_ready(bool* ready) override;

    /**
    * @brief Checks if the given model is ready using HTTP protocal
    */
    triton::client::Error is_model_ready(bool* ready, std::string& model_name) override;

    /**
    * @brief Gets the config for the given model using HTTP protocal
    */
    triton::client::Error model_config(std::string* model_config, std::string& model_name) override;

    /**
    * @brief Gets metadata for the given model using HTTP protocal
    */
    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override;

    /**
    * @brief Runs Triton Server inference given the model options, inputs, and outputs, using HTTP protocal
    */
    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<TritonInferInput>& inputs,
                                      const std::vector<TritonInferRequestedOutput>& outputs) override;
};

class MORPHEUS_EXPORT TritonInferenceClientSession : public IInferenceClientSession
{
  private:
    std::string m_model_name;
    TensorIndex m_max_batch_size = -1;
    std::vector<TritonInOut> m_model_inputs;
    std::vector<TritonInOut> m_model_outputs;
    std::shared_ptr<ITritonClient> m_client;

  public:
    TritonInferenceClientSession(std::shared_ptr<ITritonClient> client, std::string model_name);

    /**
      @brief Gets the inference input mappings for Triton
    */
    std::vector<TensorModelMapping> get_input_mappings(std::vector<TensorModelMapping> input_map_overrides) override;

    /**
      @brief Gets the inference output mappings for Triton
    */
    std::vector<TensorModelMapping> get_output_mappings(std::vector<TensorModelMapping> output_map_overrides) override;

    /**
      @brief Invokes a single tensor inference using the constructor-provided ITritonClient
    */
    mrc::coroutines::Task<TensorMap> infer(TensorMap&& inputs) override;
};

class MORPHEUS_EXPORT TritonInferenceClient : public IInferenceClient
{
  private:
    std::shared_ptr<ITritonClient> m_client;
    std::string m_model_name;
    std::shared_ptr<TritonInferenceClientSession> m_session;

  public:
    TritonInferenceClient(std::unique_ptr<ITritonClient>&& client, std::string model_name);

    /**
      @brief Gets or creates a TritonInferenceClientSession
    */
    std::shared_ptr<IInferenceClientSession> get_session() override;

    /**
      @brief Resets the inference session pointer so it will be recreated on the next call to get_session().
    */
    void reset_session() override;
};

}  // namespace morpheus
