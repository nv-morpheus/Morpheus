# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import typing
from functools import partial

import cupy as cp
import mrc
import numpy as np

import cudf

import morpheus._lib.messages as _messages
import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.cli.utils import MorpheusRelativePath
from morpheus.cli.utils import get_package_relative_file
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage
from morpheus.utils.cudf_subword_helper import tokenize_text_series

logger = logging.getLogger(__name__)


def cupyarray_to_base64(cupy_array):
    array_bytes = cupy_array.get().tobytes()
    array_shape = cupy_array.shape
    array_dtype = str(cupy_array.dtype)

    # Create a dictionary to store bytes, shape, and dtype
    encoded_dict = {'bytes': base64.b64encode(array_bytes).decode("utf-8"), 'shape': array_shape, 'dtype': array_dtype}

    # Convert dictionary to JSON string for storage
    return json.dumps(encoded_dict)


def base64_to_cupyarray(base64_str):
    # Convert JSON string back to dictionary
    encoded_dict = json.loads(base64_str)

    # Extract bytes, shape, and dtype
    array_bytes = base64.b64decode(encoded_dict['bytes'])
    array_shape = tuple(encoded_dict['shape'])
    array_dtype = encoded_dict['dtype']

    # Convert bytes back to a NumPy array and reshape
    np_array = np.frombuffer(array_bytes, dtype=array_dtype).reshape(array_shape)

    # Convert NumPy array to CuPy array
    cp_array = cp.array(np_array)

    return cp_array


@register_stage(
    "preprocess",
    modes=[PipelineModes.NLP],
    option_args={"vocab_hash_file": {
        "type": MorpheusRelativePath(exists=True, dir_okay=False, resolve_path=True)
    }})
class PreprocessNLPStage(PreprocessBaseStage):
    """
    Prepare NLP input DataFrames for inference.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    vocab_hash_file : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function.
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    add_special_tokens : bool
        Whether to encode the sequences with the special tokens of the BERT classification model.
    stride : int
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.
    column : str
        Name of the column containing the data that needs to be preprocessed.

    """

    def __init__(self,
                 c: Config,
                 vocab_hash_file: str = "data/bert-base-cased-hash.txt",
                 truncation: bool = False,
                 do_lower_case: bool = False,
                 add_special_tokens: bool = False,
                 stride: int = -1,
                 column: str = "data"):
        super().__init__(c)

        self._column = column
        self._seq_length = c.feature_length
        self._vocab_hash_file = get_package_relative_file(vocab_hash_file)

        if (stride <= 0):
            # Set the stride to 75%. Works well with powers of 2
            self._stride = self._seq_length // 2
            self._stride = self._stride + self._stride // 2
        else:
            # Use the given value
            self._stride = stride

        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens

    @property
    def name(self) -> str:
        return "preprocess-nlp"

    def supports_cpp_node(self):
        return True

    @staticmethod
    def pre_process_batch(message: typing.Union[MultiMessage, ControlMessage],
                          vocab_hash_file: str,
                          do_lower_case: bool,
                          seq_len: int,
                          stride: int,
                          truncation: bool,
                          add_special_tokens: bool,
                          column: str) -> typing.Union[MultiInferenceNLPMessage, ControlMessage]:
        """
        For NLP category use cases, this function performs pre-processing.

        [parameters are the same as the original function]

        Returns
        -------
        `morpheus.pipeline.messages.MultiInferenceNLPMessage`
            NLP inference message.

        """
        if isinstance(message, ControlMessage):
            return PreprocessNLPStage.process_control_message(message,
                                                              vocab_hash_file,
                                                              do_lower_case,
                                                              seq_len,
                                                              stride,
                                                              truncation,
                                                              add_special_tokens,
                                                              column)
        if isinstance(message, MultiMessage):
            return PreprocessNLPStage.process_multi_message(message,
                                                            vocab_hash_file,
                                                            do_lower_case,
                                                            seq_len,
                                                            stride,
                                                            truncation,
                                                            add_special_tokens,
                                                            column)

        raise TypeError("Unsupported message type")

    @staticmethod
    def process_control_message(message: ControlMessage,
                                vocab_hash_file: str,
                                do_lower_case: bool,
                                seq_len: int,
                                stride: int,
                                truncation: bool,
                                add_special_tokens: bool,
                                column: str) -> ControlMessage:

        with message.payload().mutable_dataframe() as mdf:
            text_series = cudf.Series(mdf[column])

        tokenized = tokenize_text_series(vocab_hash_file=vocab_hash_file,
                                         do_lower_case=do_lower_case,
                                         text_ser=text_series,
                                         seq_len=seq_len,
                                         stride=stride,
                                         truncation=truncation,
                                         add_special_tokens=add_special_tokens)

        del text_series

        # We need the C++ impl of TensorMemory until #1646 is resolved
        message.tensors(
            _messages.TensorMemory(count=tokenized.input_ids.shape[0],
                                   tensors={
                                       "input_ids": tokenized.input_ids,
                                       "input_mask": tokenized.input_mask,
                                       "seq_ids": tokenized.segment_ids
                                   }))

        message.set_metadata("inference_memory_params", {"inference_type": "nlp"})
        return message

    @staticmethod
    def process_multi_message(message: MultiMessage,
                              vocab_hash_file: str,
                              do_lower_case: bool,
                              seq_len: int,
                              stride: int,
                              truncation: bool,
                              add_special_tokens: bool,
                              column: str) -> MultiInferenceNLPMessage:
        # Existing logic for MultiMessage
        text_ser = cudf.Series(message.get_meta(column))

        tokenized = tokenize_text_series(vocab_hash_file=vocab_hash_file,
                                         do_lower_case=do_lower_case,
                                         text_ser=text_ser,
                                         seq_len=seq_len,
                                         stride=stride,
                                         truncation=truncation,
                                         add_special_tokens=add_special_tokens)
        del text_ser

        seg_ids = tokenized.segment_ids
        seg_ids[:, 0] = seg_ids[:, 0] + message.mess_offset

        memory = InferenceMemoryNLP(count=tokenized.input_ids.shape[0],
                                    input_ids=tokenized.input_ids,
                                    input_mask=tokenized.input_mask,
                                    seq_ids=seg_ids)

        infer_message = MultiInferenceNLPMessage.from_message(message, memory=memory)

        return infer_message

    def _get_preprocess_fn(
        self
    ) -> typing.Callable[[typing.Union[MultiMessage, ControlMessage]],
                         typing.Union[MultiInferenceMessage, ControlMessage]]:
        return partial(PreprocessNLPStage.pre_process_batch,
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens,
                       column=self._column)

    def _get_preprocess_node(self, builder: mrc.Builder):
        if (self._use_control_message):
            return _stages.PreprocessNLPControlMessageStage(builder,
                                                            self.unique_name,
                                                            self._vocab_hash_file,
                                                            self._seq_length,
                                                            self._truncation,
                                                            self._do_lower_case,
                                                            self._add_special_tokens,
                                                            self._stride,
                                                            self._column)

        return _stages.PreprocessNLPMultiMessageStage(builder,
                                                      self.unique_name,
                                                      self._vocab_hash_file,
                                                      self._seq_length,
                                                      self._truncation,
                                                      self._do_lower_case,
                                                      self._add_special_tokens,
                                                      self._stride,
                                                      self._column)
