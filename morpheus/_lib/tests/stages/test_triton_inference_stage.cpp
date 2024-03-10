
#include "../test_utils/common.hpp"  // IWYU pragma: associated
#include "common.h"
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
#include "morpheus/utilities/table_util.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/test_scheduler.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>

class FakeInferResult : public triton::client::InferResult
{
    triton::client::Error RequestStatus() const override
    {
        throw std::runtime_error("not implemented");
    }

    std::string DebugString() const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error Id(std::string* id) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error ModelName(std::string* name) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error ModelVersion(std::string* version) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error Shape(const std::string& output_name, std::vector<int64_t>* shape) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error Datatype(const std::string& output_name, std::string* datatype) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error StringData(const std::string& output_name,
                                     std::vector<std::string>* string_result) const override
    {
        throw std::runtime_error("not implemented");
    }

    triton::client::Error RawData(const std::string& output_name, const uint8_t** buf, size_t* byte_size) const override
    {
        throw std::runtime_error("not implemented");
    }
};

class FakeTritonClient : public morpheus::ITritonClient
{
  private:
    std::shared_ptr<mrc::coroutines::Scheduler> m_on;

  public:
    FakeTritonClient(std::shared_ptr<mrc::coroutines::Scheduler> on) : m_on(std::move(on)) {}

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
        *model_config = R"({
            "max_batch_size": 100
        })";

        return triton::client::Error::Success;
    }

    triton::client::Error model_metadata(std::string* model_metadata, std::string& model_name) override
    {
        *model_metadata = R"({
            "inputs":[
                {
                    "name":"seq_ids",
                    "shape": [0, 1],
                    "datatype":"INT32"
                }
            ],
            "outputs":[
                {
                    "name":"seq_ids",
                    "shape": [0, 1],
                    "datatype":"INT32"
                }
            ]})";

        return triton::client::Error::Success;
    }

    triton::client::Error async_infer(triton::client::InferenceServerHttpClient::OnCompleteFn callback,
                                      const triton::client::InferOptions& options,
                                      const std::vector<triton::client::InferInput*>& inputs,
                                      const std::vector<const triton::client::InferRequestedOutput*>& outputs) override
    {
        callback(new FakeInferResult());

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

cudf::io::table_with_metadata create_test_table_with_metadata(uint32_t rows)
{
    cudf::data_type cudf_data_type{cudf::type_to_id<int>()};

    auto column = cudf::make_fixed_width_column(cudf_data_type, rows);

    std::vector<int> data(rows);
    std::iota(data.begin(), data.end(), 0);

    cudaMemcpy(column->mutable_view().data<int>(),
               data.data(),
               data.size() * sizeof(int),
               cudaMemcpyKind::cudaMemcpyHostToDevice);

    std::vector<std::unique_ptr<cudf::column>> columns;

    columns.emplace_back(std::move(column));

    auto table = std::make_unique<cudf::table>(std::move(columns));

    auto index_info   = cudf::io::column_name_info{""};
    auto column_names = std::vector<cudf::io::column_name_info>({{index_info}});
    auto metadata     = cudf::io::table_metadata{std::move(column_names), {}, {}};

    return cudf::io::table_with_metadata{std::move(table), metadata};
}

TEST_F(TestTritonInferenceStage, SingleRow)
{
    cudf::data_type cudf_data_type{cudf::type_to_id<int>()};

    auto column = cudf::make_fixed_width_column(cudf_data_type, 1);

    column->mutable_view().data<int>();

    const std::size_t count = 10;
    const auto dtype        = morpheus::DType::create<int>();
    auto buffer = std::make_shared<rmm::device_buffer>(count * dtype.item_size(), rmm::cuda_stream_per_thread);
    std::vector<int> seq_ids({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    cudaMemcpy(buffer->data(), seq_ids.data(), count * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    auto tensors = morpheus::TensorMap();
    tensors["seq_ids"].swap(morpheus::Tensor::create(buffer, dtype, {count, 1}, {}));
    auto memory = std::make_shared<morpheus::TensorMemory>(count, std::move(tensors));

    // auto table           = morpheus::load_table_from_file(input_file);
    auto table = create_test_table_with_metadata(count);
    // auto index_col_count = morpheus::prepare_df_index(table);
    auto meta    = morpheus::MessageMeta::create_from_cpp(std::move(table), 1);
    auto message = std::make_shared<morpheus::MultiInferenceMessage>(meta, 0, count, memory);

    auto on            = std::make_shared<mrc::coroutines::TestScheduler>();
    auto create_client = [&]() {
        return std::make_unique<FakeTritonClient>(on);
    };
    auto stage = morpheus::InferenceClientStage(create_client, "", false, {}, {});

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