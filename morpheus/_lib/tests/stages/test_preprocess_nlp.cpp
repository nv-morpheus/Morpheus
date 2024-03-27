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

// TEST_CLASS(PreprocessNLP);

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

TEST_F(TestPreprocessNLP, GetIndexColCountNoIdxFromFile)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "countries.csv";

    auto test_vocab_hash_file_dir = test::get_morpheus_root() / "morpheus/data";
    std::filesystem::path vocab_hash_file = test_vocab_hash_file_dir / "bert-base-cased-hash.txt";

    auto msg = std::make_shared<ControlMessage>();

    // Create a dataframe from a file
    auto table = load_table_from_file(input_file);
    auto payload = MessageMeta::create_from_cpp(std::move(table));

    // Set the dataframe as the payload
    msg->payload(payload);

    pybind11::gil_scoped_release no_gil;
    // Create the stage (Passing in the requried parameters)
    auto stage = std::make_shared<PreprocessNLPStage<ControlMessage, ControlMessage>>(vocab_hash_file /*vocab_hash_file*/,
                                                                                      1 /*sequence_length*/,
                                                                                      false /*truncation*/,
                                                                                      false /*do_lower_case*/,
                                                                                      false /*add_special_token*/,
                                                                                      1 /*stride*/,
                                                                                      "country" /*column*/);

    // Call the process method to handle one message
    auto response = stage->pre_process_batch(msg);

    // Validate response here
    auto response_payload = response->payload();
    
    std::cout << response_payload->count() << std::endl;
}
