
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/stages/triton_inference.hpp"

#include <gtest/gtest.h>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/test_scheduler.hpp>

#include <iostream>

class TestTritonInferenceStage : public ::testing::Test
{};

TEST_F(TestTritonInferenceStage, OnData)
{
    auto stage = morpheus::InferenceClientStage("", "", false, false, false, {}, {});

    std::cout << "checkpoint 1" << std::endl;

    auto on = std::make_shared<mrc::coroutines::TestScheduler>();

    std::cout << "checkpoint 2" << std::endl;

    auto message = std::make_shared<morpheus::MultiInferenceMessage>();

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