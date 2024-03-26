#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace morpheus;

TEST_CLASS(PreprocessNLP);

TEST_F(TestPreprocessNLP, GetIndexColCountNoIdxFromFile)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    std::filesystem::path input_file = test_data_dir / "filter_probs.csv";

    auto msg = std::make_shared<ControlMessage>();

    // Create a dataframe from a file
    auto payload = MessageMeta::create_from_cpp(load_table_from_file(input_file));

    // Set the dataframe as the payload
    msg->payload(payload);

    // Create the stage (Passing in the requried parameters)
    auto stage = std::make_shared<PreprocessNLPStage<ControlMessage, ControlMessage>>("" /*vocab_hash_file*/,
                                                                                      0 /*sequence_length*/,
                                                                                      false /*truncation*/,
                                                                                      false /*do_lower_case*/,
                                                                                      false /*add_special_token*/,
                                                                                      0 /*stride*/,
                                                                                      "" /*column*/);

    // Call the process method to handle one message
    auto response = stage->pre_process_batch(msg);

    // Validate response here
}
