#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/add_classification.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>

using namespace morpheus;

TEST_CLASS(AddClassification);

TEST_F(TestAddClassification, TestProcessControlMessageAndMultiResponseMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "bools.csv";

    TensorIndex cols_size  = 3;
    TensorIndex mess_count = 3;
    auto packed_data =
        std::make_shared<rmm::device_buffer>(cols_size * mess_count * sizeof(double), rmm::cuda_stream_per_thread);

    cudf::io::csv_reader_options read_opts = cudf::io::csv_reader_options::builder(cudf::io::source_info(input_file))
                                                 .dtypes({cudf::data_type(cudf::data_type{cudf::type_to_id<bool>()})})
                                                 .header(0);
    cudf::io::table_with_metadata table_with_meta = cudf::io::read_csv(read_opts);
    auto meta = MessageMeta::create_from_cpp(std::move(table_with_meta));

    std::map<std::size_t, std::string> idx2label = {{0, "bool"}};

    // Create MultiResponseMessage
    auto tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto tensor_memory = std::make_shared<TensorMemory>(mess_count);
    tensor_memory->set_tensor("probs", std::move(tensor));
    auto multi = std::make_shared<MultiResponseMessage>(meta, 0, mess_count, std::move(tensor_memory));

    // Create PreProcessMultiMessageStage
    auto multi_stage =
        std::make_shared<AddClassificationsStage<MultiResponseMessage, MultiResponseMessage>>(idx2label, 0.0);
    auto multi_response              = multi_stage->pre_process_batch(multi);
    auto multi_response_probs_tensor = multi_response->get_tensor(multi_response->probs_tensor_name);

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);
    auto cm_tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto cm_tensor_memory = std::make_shared<TensorMemory>(mess_count);
    cm_tensor_memory->set_tensor("probs", std::move(cm_tensor));
    cm->tensors(cm_tensor_memory);

    // Create PreProcessControlMessageStage
    auto cm_stage = std::make_shared<AddClassificationsStage<ControlMessage, ControlMessage>>(idx2label, 0.0);

    auto cm_response              = cm_stage->pre_process_batch(cm);
    auto cm_response_probs_tensor = cm_response->tensors()->get_tensor("probs");

    // Check the returned tensors have the same size
    EXPECT_EQ(multi_response_probs_tensor.count(), cm_response_probs_tensor.count());
}
