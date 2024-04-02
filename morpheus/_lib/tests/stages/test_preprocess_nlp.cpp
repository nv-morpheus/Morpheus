#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <gtest/gtest.h>

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
    auto cm_response_payload = cm_response->payload();
    EXPECT_EQ(cm_response_payload->count(), 193);

    // Create MultiMessage
    auto multi = std::make_shared<MultiMessage>(meta);

    // Create PreProcessMultiMessageStage
    auto multi_stage            = std::make_shared<PreprocessNLPStageMM>(vocab_hash_file /*vocab_hash_file*/,
                                                              1 /*sequence_length*/,
                                                              false /*truncation*/,
                                                              false /*do_lower_case*/,
                                                              false /*add_special_token*/,
                                                              1 /*stride*/,
                                                              "country" /*column*/);
    auto multi_response         = multi_stage->on_data(multi);
    auto multi_response_payload = multi_response->meta;

    // Check if identical number of rows are returned
    EXPECT_EQ(multi_response_payload->count(), cm_response_payload->count());
}
