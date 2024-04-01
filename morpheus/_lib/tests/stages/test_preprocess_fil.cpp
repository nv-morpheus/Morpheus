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
    auto cm_stage =
        std::make_shared<PreprocessFILStage<ControlMessage, ControlMessage>>(std::vector<std::string>{"country"});

    auto cm_response         = cm_stage->on_data(cm);
    auto cm_response_payload = cm_response->payload();
    EXPECT_EQ(cm_response_payload->count(), 193);

    // Create MultiMessage
    auto multi = std::make_shared<MultiMessage>(meta);
    // Create PreProcessMultiMessageStage
    auto multi_stage =
        std::make_shared<PreprocessFILStage<MultiMessage, MultiInferenceMessage>>(std::vector<std::string>{"country"});
    auto multi_response         = multi_stage->on_data(multi);
    auto multi_response_payload = multi_response->meta;

    // Check if identical number of rows are returned
    EXPECT_EQ(multi_response_payload->count(), cm_response_payload->count());
}
