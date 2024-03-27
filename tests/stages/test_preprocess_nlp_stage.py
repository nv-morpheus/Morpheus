import pytest
import cudf
import cupy as cp

from unittest.mock import patch, Mock
from morpheus._lib.messages import ControlMessage, MultiMessage
from morpheus.config import Config
from morpheus.messages import MessageMeta

from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    config.edge_buffer_size = 4
    config.feature_length = 256
    config.mode = "NLP"
    config.model_max_batch_size = 32
    config.num_threads = 1
    config.pipeline_batch_size = 64
    yield config


def test_constructor(config: Config):
    stage = PreprocessNLPStage(config)
    assert stage.name == "preprocess-nlp"
    assert stage._column == "data"
    assert stage._seq_length == 256
    assert stage._vocab_hash_file.endswith("data/bert-base-cased-hash.txt")
    assert stage._truncation == False
    assert stage._do_lower_case == False
    assert stage._add_special_tokens == False

    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


@patch("morpheus.stages.preprocess.preprocess_nlp_stage.tokenize_text_series")
def test_process_control_message(mock_tokenize_text_series, config: Config):
    mock_tokenized = Mock()
    mock_tokenized.input_ids = cp.array([[1, 2], [1, 2]])
    mock_tokenized.input_mask = cp.array([[3, 4], [3, 4]])
    mock_tokenized.segment_ids = cp.array([[0, 0], [1, 1]])
    mock_tokenize_text_series.return_value = mock_tokenized

    stage = PreprocessNLPStage(config)
    input_cm = ControlMessage()
    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)
    input_cm.payload(meta)

    output_cm = stage.pre_process_batch(input_cm,
                                        stage._vocab_hash_file,
                                        stage._do_lower_case,
                                        stage._seq_length,
                                        stage._stride,
                                        stage._truncation,
                                        stage._add_special_tokens,
                                        stage._column)
    assert output_cm.get_metadata("inference_memory_params") == {"inference_type": "nlp"}
    assert cp.array_equal(output_cm.tensors().get_tensor("input_ids"), mock_tokenized.input_ids)
    assert cp.array_equal(output_cm.tensors().get_tensor("input_mask"), mock_tokenized.input_mask)
    assert cp.array_equal(output_cm.tensors().get_tensor("seq_ids"), mock_tokenized.segment_ids)


@patch("morpheus.stages.preprocess.preprocess_nlp_stage.tokenize_text_series")
def test_process_multi_message(mock_tokenize_text_series, config: Config):
    mock_tokenized = Mock()
    mock_tokenized.input_ids = cp.array([[1, 2], [1, 2]])
    mock_tokenized.input_mask = cp.array([[3, 4], [3, 4]])
    mock_tokenized.segment_ids = cp.array([[0, 0], [1, 1]])
    mock_tokenize_text_series.return_value = mock_tokenized

    stage = PreprocessNLPStage(config)
    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)
    mess_offset = 0
    input_multi_message = MultiMessage(meta=meta, mess_offset=mess_offset, mess_count=2)

    output_infer_message = stage.pre_process_batch(input_multi_message,
                                                   stage._vocab_hash_file,
                                                   stage._do_lower_case,
                                                   stage._seq_length,
                                                   stage._stride,
                                                   stage._truncation,
                                                   stage._add_special_tokens,
                                                   stage._column)
    assert cp.array_equal(output_infer_message.input_ids, mock_tokenized.input_ids)
    assert cp.array_equal(output_infer_message.input_mask, mock_tokenized.input_mask)
    mock_tokenized.segment_ids[:, 0] = mock_tokenized.segment_ids[:, 0] + mess_offset
    assert cp.array_equal(output_infer_message.seq_ids, mock_tokenized.segment_ids)


@patch("morpheus.stages.preprocess.preprocess_nlp_stage.tokenize_text_series")
def test_process_control_message_and_multi_message(mock_tokenize_text_series, config: Config):
    mock_tokenized = Mock()
    mock_tokenized.input_ids = cp.array([[1, 2], [1, 2]])
    mock_tokenized.input_mask = cp.array([[3, 4], [3, 4]])
    mock_tokenized.segment_ids = cp.array([[0, 0], [1, 1]])
    mock_tokenize_text_series.return_value = mock_tokenized

    stage = PreprocessNLPStage(config)
    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)
    input_control_message = ControlMessage()
    input_control_message.payload(meta)

    mess_offset = 0
    input_multi_message = MultiMessage(meta=meta, mess_offset=mess_offset, mess_count=2)

    output_control_message = stage.pre_process_batch(input_control_message,
                                        stage._vocab_hash_file,
                                        stage._do_lower_case,
                                        stage._seq_length,
                                        stage._stride,
                                        stage._truncation,
                                        stage._add_special_tokens,
                                        stage._column)

    output_infer_message = stage.pre_process_batch(input_multi_message,
                                                   stage._vocab_hash_file,
                                                   stage._do_lower_case,
                                                   stage._seq_length,
                                                   stage._stride,
                                                   stage._truncation,
                                                   stage._add_special_tokens,
                                                   stage._column)

    # Check if each tensor in the control message is equal to the corresponding tensor in the inference message
    for tensor_key in output_control_message.tensors().tensor_names:
        assert cp.array_equal(output_control_message.tensors().get_tensor(tensor_key), getattr(output_infer_message, tensor_key))
