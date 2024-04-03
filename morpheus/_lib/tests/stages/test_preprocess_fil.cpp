#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/preprocess_fil.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace morpheus;

TEST_CLASS(PreprocessFIL);

TEST_F(TestPreprocessFIL, TestProcessControlMessageAndMultiMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "float_str.csv";

    // Create a dataframe from a file
    auto cm_table = load_table_from_file(input_file);
    auto cm_meta  = MessageMeta::create_from_cpp(std::move(cm_table));

    auto mm_table = load_table_from_file(input_file);
    auto mm_meta  = MessageMeta::create_from_cpp(std::move(mm_table));

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(cm_meta);

    // Create PreProcessControlMessageStage
    auto cm_stage = std::make_shared<PreprocessFILStageCC>(std::vector<std::string>{"float_str1", "float_str2"});
    auto cm_response = cm_stage->on_data(cm);

    // Create MultiMessage
    auto mm = std::make_shared<MultiMessage>(mm_meta);
    // Create PreProcessMultiMessageStage
    auto mm_stage    = std::make_shared<PreprocessFILStageMM>(std::vector<std::string>{"float_str1", "float_str2"});
    auto mm_response = mm_stage->on_data(mm);

    auto cm_tensors = cm_response->tensors();
    auto mm_tensors = mm_response->memory;

    // Verify output tensors
    std::vector<float> expected_input__0 = {1, 4, 2, 5, 3, 6};
    auto cm_input__0 = cm_tensors->get_tensor("input__0");
    auto mm_input__0 = mm_tensors->get_tensor("input__0");
    std::vector<float> cm_input__0_host(cm_input__0.count());
    std::vector<float> mm_input__0_host(mm_input__0.count());
    MRC_CHECK_CUDA(cudaMemcpy(cm_input__0_host.data(), cm_input__0.data(), cm_input__0.count() * sizeof(float), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(mm_input__0_host.data(), mm_input__0.data(), mm_input__0.count() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected_input__0, cm_input__0_host);
    EXPECT_EQ(cm_input__0_host, mm_input__0_host);

    // Col1 in MatxUtil__MatxCreateSegIds is not initialized
    // auto cm_seq_ids = cm_tensors->get_tensor("seq_ids");
    // auto mm_seq_ids = mm_tensors->get_tensor("seq_ids");
    // std::vector<TensorIndex> cm_seq_ids_host(cm_seq_ids.count());
    // std::vector<TensorIndex> mm_seq_ids_host(mm_seq_ids.count());
    // MRC_CHECK_CUDA(cudaMemcpy(cm_seq_ids_host.data(), cm_seq_ids.data(), cm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    // MRC_CHECK_CUDA(cudaMemcpy(mm_seq_ids_host.data(), mm_seq_ids.data(), mm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    // EXPECT_EQ(cm_seq_ids_host, mm_seq_ids_host);
}
