
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/stages/triton_inference.hpp"

#include <gtest/gtest.h>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/test_scheduler.hpp>

#include <iostream>
#include <memory>

class FakeTritonClient : public morpheus::ITritonClient
{
  public:
    triton::client::Error is_server_live(bool* live) override
    {
        *live = true;

        return triton::client::Error::Success;
    }

    triton::client::Error is_server_ready(bool* ready) override
    {
        *ready = true;

        return triton::client::Error::Success;
    }

    triton::client::Error is_model_ready(bool* ready, std::string& model_name) override
    {
        *ready = true;

        return triton::client::Error::Success;
    }

    triton::client::Error model_config(std::string* model_config, std::string& model_name) override
    {
        *model_config = R"({})";

        return triton::client::Error::Success;
    }

    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override
    {
        *model_metadata = R"({"inputs":[],"outputs":[]})";

        return triton::client::Error::Success;
    }

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<triton::client::InferInput*>& inputs,
                                      const std::vector<const triton::client::InferRequestedOutput*>& outputs =
                                          std::vector<const triton::client::InferRequestedOutput*>()) override
    {
        return triton::client::Error::Success;
    }
};

class TestTritonInferenceStage : public ::testing::Test
{};

TEST_F(TestTritonInferenceStage, OnData)
{
    auto create_client = []() {
        return std::make_unique<FakeTritonClient>();
    };

    auto stage = morpheus::InferenceClientStage(create_client, "", false, {}, {});

    auto on = std::make_shared<mrc::coroutines::TestScheduler>();

    auto message = std::shared_ptr<morpheus::MultiInferenceMessage>();

    auto results_task = [](auto& stage, auto message, auto on)
        -> mrc::coroutines::Task<std::vector<std::shared_ptr<morpheus::MultiResponseMessage>>> {
        std::vector<std::shared_ptr<morpheus::MultiResponseMessage>> results;

        auto responses_generator = stage.on_data(std::move(message), on);

        auto iter = co_await responses_generator.begin();

        while (iter != responses_generator.end())
        {
            results.emplace_back(std::move(*iter));

            co_await ++iter;
        }

        co_return results;
    }(stage, message, on);

    results_task.resume();

    while (on->resume_next()) {}

    ASSERT_THROW(results_task.promise().result(), std::runtime_error);
}