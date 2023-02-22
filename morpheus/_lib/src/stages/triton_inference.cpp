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

#include "morpheus/stages/triton_inference.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"  // for InferenceMemory
#include "morpheus/messages/memory/response_memory.hpp"   // for ResponseMemory
#include "morpheus/messages/memory/tensor_memory.hpp"     // for TensorMemory::tensor_map_t
#include "morpheus/messages/multi_response.hpp"           // for MultiResponseMessage
#include "morpheus/objects/dev_mem_info.hpp"              // for DevMemInfo
#include "morpheus/objects/dtype.hpp"                     // for DType
#include "morpheus/objects/rmm_tensor.hpp"                // for RMMTensor
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject
#include "morpheus/objects/triton_in_out.hpp"
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/stage_util.hpp"   // for foreach_map
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR
#include "morpheus/utilities/tensor_util.hpp"  // for get_elem_count

#include <bits/c++config.h>
#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice
#include <glog/logging.h>
#include <http_client.h>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <nlohmann/json.hpp>
#include <pymrc/node.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <algorithm>  // for min
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>    // for runtime_error, out_of_range
#include <type_traits>  // for declval
#include <utility>
// IWYU pragma: no_include <initializer_list>

#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

namespace {

using namespace morpheus;
using tensor_map_t = TensorMemory::tensor_map_t;
using buffer_map_t = std::map<std::string, std::shared_ptr<rmm::device_buffer>>;

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

void build_output_tensors(std::size_t count,
                          const std::vector<TritonInOut>& model_outputs,
                          buffer_map_t& output_buffers,
                          tensor_map_t& output_tensors)
{
    // Create the output memory blocks
    for (auto& model_output : model_outputs)
    {
        std::vector<TensorIndex> total_shape{model_output.shape.begin(), model_output.shape.end()};

        // First dimension will always end up being the number of rows in the dataframe
        total_shape[0]  = static_cast<TensorIndex>(count);
        auto elem_count = TensorUtils::get_elem_count(total_shape);

        // Create the output memory
        auto output_buffer = std::make_shared<rmm::device_buffer>(elem_count * model_output.datatype.item_size(),
                                                                  rmm::cuda_stream_per_thread);

        output_buffers[model_output.mapped_name] = output_buffer;

        // Triton results are always in row-major as required by the KServe protocol
        // https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data
        std::vector<TensorIndex> stride{total_shape[1], 1};
        output_tensors[model_output.mapped_name] =
            Tensor::create(std::move(output_buffer), model_output.datatype, total_shape, stride, 0);
    }
}

std::vector<int32_t> get_seq_ids(const InferenceClientStage::sink_type_t& message)
{
    // Take a copy of the sequence Ids allowing us to map rows in the response to rows in the dataframe
    // The output tensors we store in `reponse_memory` will all be of the same length as the the
    // dataframe. seq_ids has three columns, but we are only interested in the first column.
    auto seq_ids         = message->get_input("seq_ids");
    const auto item_size = seq_ids.dtype().item_size();

    std::vector<int32_t> host_seq_ids(message->count);
    MRC_CHECK_CUDA(cudaMemcpy2D(host_seq_ids.data(),
                                item_size,
                                seq_ids.data(),
                                seq_ids.stride(0) * item_size,
                                item_size,
                                host_seq_ids.size(),
                                cudaMemcpyDeviceToHost));

    return host_seq_ids;
}

std::pair<std::shared_ptr<triton::client::InferInput>, std::vector<uint8_t>> build_input(
    const InferenceClientStage::sink_type_t& msg_slice, const TritonInOut& model_input)
{
    DCHECK(msg_slice->memory->has_tensor(model_input.mapped_name))
        << "Model input '" << model_input.mapped_name << "' not found in InferenceMemory";

    auto const& inp_tensor = msg_slice->get_input(model_input.mapped_name);

    // Convert to the right type. Make shallow if necessary
    auto final_tensor = inp_tensor.as_type(model_input.datatype);

    std::vector<uint8_t> inp_data = final_tensor.get_host_data();

    // Test
    triton::client::InferInput* inp_ptr;

    triton::client::InferInput::Create(
        &inp_ptr, model_input.name, {inp_tensor.shape(0), inp_tensor.shape(1)}, model_input.datatype.triton_str());

    std::shared_ptr<triton::client::InferInput> inp_shared;
    inp_shared.reset(inp_ptr);

    inp_ptr->AppendRaw(inp_data);

    return std::make_pair(inp_shared, std::move(inp_data));
}

std::shared_ptr<const triton::client::InferRequestedOutput> build_output(const TritonInOut& model_output)
{
    triton::client::InferRequestedOutput* out_ptr;

    triton::client::InferRequestedOutput::Create(&out_ptr, model_output.name);
    std::shared_ptr<const triton::client::InferRequestedOutput> out_shared;
    out_shared.reset(out_ptr);

    return out_shared;
}

void reduce_outputs(const InferenceClientStage::sink_type_t& x,
                    buffer_map_t& output_buffers,
                    tensor_map_t& output_tensors)
{
    // When our tensor lengths are longer than our dataframe we will need to use the seq_ids array to
    // lookup how the values should map back into the dataframe.
    auto host_seq_ids = get_seq_ids(x);

    tensor_map_t reduced_outputs;

    for (const auto& output : output_tensors)
    {
        DCHECK(std::dynamic_pointer_cast<RMMTensor>(output.second.get_tensor()) != nullptr);
        auto tensor = std::static_pointer_cast<RMMTensor>(output.second.get_tensor());

        const auto rank = tensor->rank();
        std::vector<TensorIndex> shape(rank);
        tensor->get_shape(shape);

        std::vector<TensorIndex> stride(rank);
        tensor->get_stride(stride);

        // DevMemInfo wants the shape & stride in size_t
        std::vector<std::size_t> tensor_shape(shape.size());
        std::copy(shape.cbegin(), shape.cend(), tensor_shape.begin());

        std::vector<std::size_t> tensor_stride(stride.size());
        std::copy(stride.cbegin(), stride.cend(), tensor_stride.begin());

        std::vector<std::size_t> reduced_shape{tensor_shape};
        reduced_shape[0] = x->mess_count;

        auto& buffer        = output_buffers[output.first];
        auto reduced_buffer = MatxUtil::reduce_max(
            DevMemInfo{buffer, tensor->dtype(), tensor_shape, tensor_stride}, host_seq_ids, 0, reduced_shape);

        output_buffers[output.first] = reduced_buffer;

        reduced_outputs[output.first] =
            Tensor::create(std::move(reduced_buffer),
                           tensor->dtype(),
                           {static_cast<TensorIndex>(reduced_shape[0]), static_cast<TensorIndex>(reduced_shape[1])},
                           stride,
                           0);
    }

    output_tensors = std::move(reduced_outputs);
}

void apply_logits(buffer_map_t& output_buffers, tensor_map_t& output_tensors)
{
    tensor_map_t logit_outputs;

    for (const auto& output : output_tensors)
    {
        DCHECK(std::dynamic_pointer_cast<RMMTensor>(output.second.get_tensor()) != nullptr);
        auto input_tensor = std::static_pointer_cast<RMMTensor>(output.second.get_tensor());

        const auto rank = input_tensor->rank();
        std::vector<TensorIndex> shape(rank);
        input_tensor->get_shape(shape);

        std::vector<TensorIndex> stride(rank);
        input_tensor->get_stride(stride);

        // DevMemInfo wants the shape & stride in size_t
        std::vector<std::size_t> input_shape(shape.size());
        std::copy(shape.cbegin(), shape.cend(), input_shape.begin());

        std::vector<std::size_t> input_stride(stride.size());
        std::copy(stride.cbegin(), stride.cend(), input_stride.begin());

        auto& buffer = output_buffers[output.first];

        auto output_buffer = MatxUtil::logits(DevMemInfo{buffer, input_tensor->dtype(), input_shape, input_stride});

        output_buffers[output.first] = output_buffer;

        // For logits the input and output shapes will be the same
        logit_outputs[output.first] = Tensor::create(std::move(output_buffer), input_tensor->dtype(), shape, stride, 0);
    }

    output_tensors = std::move(logit_outputs);
}

}  // namespace

namespace morpheus {
// Component public implementations
// ************ InferenceClientStage ************************* //
InferenceClientStage::InferenceClientStage(std::string model_name,
                                           std::string server_url,
                                           bool force_convert_inputs,
                                           bool use_shared_memory,
                                           bool needs_logits,
                                           std::map<std::string, std::string> inout_mapping) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_model_name(std::move(model_name)),
  m_server_url(std::move(server_url)),
  m_force_convert_inputs(force_convert_inputs),
  m_use_shared_memory(use_shared_memory),
  m_needs_logits(needs_logits),
  m_inout_mapping(std::move(inout_mapping)),
  m_options(m_model_name)
{
    // Connect with the server to setup the inputs/outputs
    this->connect_with_server();  // TODO(Devin)
}

InferenceClientStage::subscribe_fn_t InferenceClientStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, m_server_url, false));

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, &client](sink_type_t x) {
                // Using the `count` which is the number of rows in the inference tensors. We will check later if this
                // doesn't match the number of rows in the dataframe (`mess_count`). This happens when the size of the
                // input is too large and needs to be broken up in chunks in the pre-process stage. When this is the
                // case we will reduce the rows in the response outputs such that we have a single response for each
                // row int he dataframe.
                tensor_map_t output_tensors;
                buffer_map_t output_buffers;
                build_output_tensors(x->count, m_model_outputs, output_buffers, output_tensors);

                for (size_t start = 0; start < x->count; start += m_max_batch_size)
                {
                    triton::client::InferInput* input1;

                    size_t stop = std::min(start + m_max_batch_size, x->count);

                    sink_type_t mini_batch_input = x->get_slice(start, stop);

                    // Iterate on the model inputs in case the model takes less than what tensors are available
                    std::vector<std::pair<std::shared_ptr<triton::client::InferInput>, std::vector<uint8_t>>>
                        saved_inputs = foreach_map(m_model_inputs, [&mini_batch_input](auto const& model_input) {
                            return (build_input(mini_batch_input, model_input));
                        });

                    std::vector<std::shared_ptr<const triton::client::InferRequestedOutput>> saved_outputs =
                        foreach_map(m_model_outputs, [](auto const& model_output) {
                            // Generate the outputs to be requested.
                            return build_output(model_output);
                        });

                    std::vector<triton::client::InferInput*> inputs =
                        foreach_map(saved_inputs, [](auto x) { return x.first.get(); });

                    std::vector<const triton::client::InferRequestedOutput*> outputs =
                        foreach_map(saved_outputs, [](auto x) { return x.get(); });

                    triton::client::InferResult* results;

                    CHECK_TRITON(client->Infer(&results, m_options, inputs, outputs));

                    for (auto& model_output : m_model_outputs)
                    {
                        std::vector<int64_t> output_shape;

                        CHECK_TRITON(results->Shape(model_output.name, &output_shape));

                        // Make sure we have at least 2 dims
                        while (output_shape.size() < 2)
                        {
                            output_shape.push_back(1);
                        }

                        const uint8_t* output_ptr = nullptr;
                        size_t output_ptr_size    = 0;
                        CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

                        auto output_tensor = output_tensors[model_output.mapped_name].slice(
                            {static_cast<cudf::size_type>(start), 0}, {static_cast<cudf::size_type>(stop), -1});

                        DCHECK_EQ(stop - start, output_shape[0]);
                        DCHECK_EQ(output_tensor.bytes(), output_ptr_size);

                        MRC_CHECK_CUDA(
                            cudaMemcpy(output_tensor.data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice));
                    }
                }

                if (x->mess_count != x->count)
                {
                    reduce_outputs(x, output_buffers, output_tensors);
                }

                // If we need to do logits, do that here
                if (m_needs_logits)
                {
                    apply_logits(output_buffers, output_tensors);
                }

                // Final output of all mini-batches
                auto response_mem = std::make_shared<ResponseMemory>(x->mess_count, std::move(output_tensors));
                auto response     = std::make_shared<MultiResponseMessage>(
                    x->meta, x->mess_offset, x->mess_count, std::move(response_mem), 0, response_mem->count);

                output.on_next(std::move(response));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

void InferenceClientStage::connect_with_server()
{
    std::string server_url = m_server_url;

    std::unique_ptr<triton::client::InferenceServerHttpClient> client;

    auto result = triton::client::InferenceServerHttpClient::Create(&client, server_url, false);

    // Now load the input/outputs for the model
    bool is_server_live = false;

    triton::client::Error status = client->IsServerLive(&is_server_live);

    if (!status.IsOk())
    {
        if (this->is_default_grpc_port(server_url))
        {
            LOG(WARNING) << "Failed to connect to Triton at '" << m_server_url
                         << "'. Default gRPC port of (8001) was detected but C++ "
                            "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

            // We are using the default gRPC port, try the default HTTP
            std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

            auto result = triton::client::InferenceServerHttpClient::Create(&unique_client, server_url, false);

            client = std::move(unique_client);

            status = client->IsServerLive(&is_server_live);
        }
        else if (status.Message().find("Unsupported protocol") != std::string::npos)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR(
                "Failed to connect to Triton at '"
                << m_server_url
                << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                   "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                   "specified the HTTP port (Default 8000)."));
        }

        if (!status.IsOk())
            throw std::runtime_error(
                MORPHEUS_CONCAT_STR("Unable to connect to Triton at '"
                                    << m_server_url << "'. Check the URL and port and ensure the server is running."));
    }

    // Save this for new clients
    m_server_url = server_url;

    if (!is_server_live)
        throw std::runtime_error("Server is not live");

    bool is_server_ready = false;
    CHECK_TRITON(client->IsServerReady(&is_server_ready));

    if (!is_server_ready)
        throw std::runtime_error("Server is not ready");

    bool is_model_ready = false;
    CHECK_TRITON(client->IsModelReady(&is_model_ready, this->m_model_name));

    if (!is_model_ready)
        throw std::runtime_error("Model is not ready");

    std::string model_metadata_json;
    CHECK_TRITON(client->ModelMetadata(&model_metadata_json, this->m_model_name));

    auto model_metadata = nlohmann::json::parse(model_metadata_json);

    std::string model_config_json;
    CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

    auto model_config = nlohmann::json::parse(model_config_json);

    if (model_config.contains("max_batch_size"))
    {
        m_max_batch_size = model_config.at("max_batch_size").get<int>();
    }

    for (auto const& input : model_metadata.at("inputs"))
    {
        auto shape = input.at("shape").get<std::vector<int>>();

        auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        std::string mapped_name = input.at("name").get<std::string>();

        if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
        {
            mapped_name = m_inout_mapping[mapped_name];
        }

        m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                             bytes,
                                             DType::from_triton(input.at("datatype").get<std::string>()),
                                             shape,
                                             mapped_name,
                                             0});
    }

    for (auto const& output : model_metadata.at("outputs"))
    {
        auto shape = output.at("shape").get<std::vector<int>>();

        auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

        size_t bytes = dtype.item_size();

        for (auto& y : shape)
        {
            if (y == -1)
            {
                y = m_max_batch_size;
            }

            bytes *= y;
        }

        std::string mapped_name = output.at("name").get<std::string>();

        if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
        {
            mapped_name = m_inout_mapping[mapped_name];
        }

        m_model_outputs.push_back(
            TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, mapped_name, 0});
    }
}

bool InferenceClientStage::is_default_grpc_port(std::string& server_url)
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

// ************ InferenceClientStageInterfaceProxy********* //
std::shared_ptr<mrc::segment::Object<InferenceClientStage>> InferenceClientStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string model_name,
    std::string server_url,
    bool force_convert_inputs,
    bool use_shared_memory,
    bool needs_logits,
    std::map<std::string, std::string> inout_mapping)
{
    auto stage = builder.construct_object<InferenceClientStage>(
        name, model_name, server_url, force_convert_inputs, use_shared_memory, needs_logits, inout_mapping);

    return stage;
}
}  // namespace morpheus
