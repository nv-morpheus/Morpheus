#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/add_scores.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>

using namespace morpheus;

TEST_CLASS(AddScores);

TEST_F(TestAddScores, TestProcessControlMessageAndMultiResponseMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "floats.csv";

    TensorIndex cols_size  = 1;
    TensorIndex mess_count = 3;
    auto packed_data =
        std::make_shared<rmm::device_buffer>(cols_size * mess_count * sizeof(double), rmm::cuda_stream_per_thread);

    // Create a dataframe from a file
    auto table = load_table_from_file(input_file);
    auto meta  = MessageMeta::create_from_cpp(std::move(table));

    std::map<std::size_t, std::string> idx2label = {{0, "float"}};

    // Create MultiResponseMessage
    auto tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto tensor_memory = std::make_shared<TensorMemory>(mess_count);
    tensor_memory->set_tensor("probs", std::move(tensor));
    auto mm = std::make_shared<MultiResponseMessage>(meta, 0, mess_count, std::move(tensor_memory));

    // Create PreProcessMultiMessageStage
    auto mm_stage    = std::make_shared<AddScoresStage<MultiResponseMessage, MultiResponseMessage>>(idx2label);
    auto mm_response = mm_stage->on_data(mm);


    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);
    auto cm_tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto cm_tensor_memory = std::make_shared<TensorMemory>(mess_count);
    cm_tensor_memory->set_tensor("probs", std::move(cm_tensor));
    cm->tensors(cm_tensor_memory);

    // Create PreProcessControlMessageStage
    auto cm_stage = std::make_shared<AddScoresStage<ControlMessage, ControlMessage>>(idx2label);
    auto cm_response              = cm_stage->on_data(cm);

    // Verify the output meta
    std::vector<float> expected_meta = {0, 0, 1.4013e-45};
    auto mm_meta = mm_response->get_meta().get_column(0);
    auto cm_meta = cm_response->payload()->get_info().get_column(0);
    std::vector<float> mm_meta_host(mm_meta.size());
    std::vector<float> cm_meta_host(cm_meta.size());
    MRC_CHECK_CUDA(cudaMemcpy(mm_meta_host.data(), mm_meta.data<double>(), mm_meta.size() * sizeof(double), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(cm_meta_host.data(), cm_meta.data<double>(), cm_meta.size() * sizeof(double), cudaMemcpyDeviceToHost));
    EXPECT_EQ(mm_meta_host, expected_meta);
    EXPECT_EQ(mm_meta_host, cm_meta_host);
}
