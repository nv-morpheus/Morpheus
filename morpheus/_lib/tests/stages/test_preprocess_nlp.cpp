#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>

#include <memory>

using namespace morpheus;

class TestPreprocessNLP : public morpheus::test::TestWithPythonInterpreter
{
  protected:
    void SetUp() override
    {
        morpheus::test::TestWithPythonInterpreter::SetUp();
        {
            pybind11::gil_scoped_acquire gil;
            CudfHelper::load();
        }
    }
};

TEST_F(TestPreprocessNLP, TestProcessControlMessageAndMultiMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "countries.csv";

    auto test_vocab_hash_file_dir         = test::get_morpheus_root() / "morpheus/data";
    std::filesystem::path vocab_hash_file = test_vocab_hash_file_dir / "bert-base-cased-hash.txt";

    // Create a dataframe from a file
    auto table = load_table_from_file(input_file);
    auto meta  = MessageMeta::create_from_cpp(std::move(table));

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);

    // Create PreProcessControlMessageStage
    auto cm_stage = std::make_shared<PreprocessNLPStageCC>(vocab_hash_file /*vocab_hash_file*/,
                                                           1 /*sequence_length*/,
                                                           false /*truncation*/,
                                                           false /*do_lower_case*/,
                                                           false /*add_special_token*/,
                                                           1 /*stride*/,
                                                           "country" /*column*/);

    auto cm_response         = cm_stage->on_data(cm);

    // Create MultiMessage
    auto mm = std::make_shared<MultiMessage>(meta);

    // Create PreProcessMultiMessageStage
    auto mm_stage            = std::make_shared<PreprocessNLPStageMM>(vocab_hash_file /*vocab_hash_file*/,
                                                              1 /*sequence_length*/,
                                                              false /*truncation*/,
                                                              false /*do_lower_case*/,
                                                              false /*add_special_token*/,
                                                              1 /*stride*/,
                                                              "country" /*column*/);
    auto mm_response         = mm_stage->on_data(mm);

    auto cm_tensors = cm_response->tensors();
    auto mm_tensors = mm_response->memory;

    // Check if the tensors are the same
    auto cm_input_ids = cm_tensors->get_tensor("input_ids");
    auto mm_input_ids = mm_tensors->get_tensor("input_ids");
    std::vector<int32_t> cm_input_ids_host(cm_input_ids.count());
    std::vector<int32_t> mm_input_ids_host(mm_input_ids.count());
    MRC_CHECK_CUDA(cudaMemcpy(cm_input_ids_host.data(), cm_input_ids.data(), cm_input_ids.count() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(mm_input_ids_host.data(), mm_input_ids.data(), mm_input_ids.count() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cm_input_ids_host, mm_input_ids_host);

    auto cm_input_mask = cm_tensors->get_tensor("input_mask");
    auto mm_input_mask = mm_tensors->get_tensor("input_mask");
    std::vector<int32_t> cm_input_mask_host(cm_input_mask.count());
    std::vector<int32_t> mm_input_mask_host(mm_input_mask.count());
    MRC_CHECK_CUDA(cudaMemcpy(cm_input_mask_host.data(), cm_input_mask.data(), cm_input_mask.count() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(mm_input_mask_host.data(), mm_input_mask.data(), mm_input_mask.count() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cm_input_mask_host, mm_input_mask_host);

    auto cm_seq_ids = cm_tensors->get_tensor("seq_ids");
    auto mm_seq_ids = mm_tensors->get_tensor("seq_ids");
    std::vector<TensorIndex> cm_seq_ids_host(cm_seq_ids.count());
    std::vector<TensorIndex> mm_seq_ids_host(mm_seq_ids.count());
    MRC_CHECK_CUDA(cudaMemcpy(cm_seq_ids_host.data(), cm_seq_ids.data(), cm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(mm_seq_ids_host.data(), mm_seq_ids.data(), mm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cm_seq_ids_host, mm_seq_ids_host);
}
