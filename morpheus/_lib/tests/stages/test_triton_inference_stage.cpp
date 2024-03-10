
#include "../test_utils/common.hpp"  // IWYU pragma: associated
#include "cuda_runtime_api.h"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/stages/triton_inference.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <gtest/gtest.h>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/test_scheduler.hpp>

#include <filesystem>
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

class TestTritonInferenceStage : public morpheus::test::TestWithPythonInterpreter
{
  protected:
    void SetUp() override
    {
        morpheus::test::TestWithPythonInterpreter::SetUp();
        {
            pybind11::gil_scoped_acquire gil;

            // Initially I ran into an issue bootstrapping cudf, I was able to work-around the issue, details in:
            // https://github.com/rapidsai/cudf/issues/12862
            morpheus::CudfHelper::load();
        }
    }
};

TEST_F(TestTritonInferenceStage, OnData)
{
    auto test_data_dir = morpheus::test::get_morpheus_root() / "tests/tests_data";

    std::filesystem::path input_file = test_data_dir / "filter_probs.csv";

    auto table           = morpheus::load_table_from_file(input_file);
    auto index_col_count = morpheus::prepare_df_index(table);

    auto create_client = []() {
        return std::make_unique<FakeTritonClient>();
    };

    auto stage = morpheus::InferenceClientStage(create_client, "", false, {}, {});

    auto on = std::make_shared<mrc::coroutines::TestScheduler>();

    auto meta = morpheus::MessageMeta::create_from_cpp(std::move(table), index_col_count);


    auto tensors = morpheus::TensorMap();

    const std::size_t count = 1;
    const auto dtype        = morpheus::DType::create<int>();
    auto buffer = std::make_shared<rmm::device_buffer>(count * dtype.item_size(), rmm::cuda_stream_per_thread);
    std::vector<int> seq_ids({0});
    cudaMemcpy(buffer->data(), seq_ids.data(), count * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    tensors["seq_ids"].swap(morpheus::Tensor::create(buffer, dtype, {1, 1}, {1, 1}));

    auto memory = std::make_shared<morpheus::TensorMemory>(count, std::move(tensors));

    auto message = std::make_shared<morpheus::MultiInferenceMessage>(meta, 0, count, memory);

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

    ASSERT_NO_THROW(results_task.promise().result());

    auto results = results_task.promise().result();

    ASSERT_EQ(results.size(), 1);
}