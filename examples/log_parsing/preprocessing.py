# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import pathlib
import string
import typing
from functools import partial

import mrc

import cudf

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage
from morpheus.utils.cudf_subword_helper import tokenize_text_series


@register_stage("log-preprocess", modes=[PipelineModes.NLP])
class PreprocessLogParsingStage(PreprocessBaseStage):
    """
    NLP usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    vocab_hashfile : pathlib.Path, exists = True, dir_okay = False
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function
    truncation : bool, default = False, is_flag = True
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool, default = False, is_flag = True
        If set to true, original text will be lowercased before encoding.
    add_special_tokens : bool, default = False, is_flag = True
        Whether or not to encode the sequences with the special tokens of the BERT classification model.
    stride : int, default = -1
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.
    column : str, default = "raw"
        Name of the column containing the data that needs to be preprocessed.

    """

    def __init__(self,
                 c: Config,
                 vocab_hash_file: pathlib.Path,
                 truncation: bool = False,
                 do_lower_case: bool = False,
                 add_special_tokens: bool = False,
                 stride: int = -1,
                 column: str = "raw"):
        super().__init__(c)

        self._seq_length = c.feature_length
        self._vocab_hash_file = vocab_hash_file

        if (stride <= 0):
            # Set the stride to 75%. Works well with powers of 2
            self._stride = self._seq_length // 2
            self._stride = self._stride + self._stride // 2
        else:
            # Use the given value
            self._stride = stride

        self._column = column
        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens

    @property
    def name(self) -> str:
        return "preprocess-logparsing"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(x: MultiMessage,
                          vocab_hash_file: str,
                          do_lower_case: bool,
                          seq_len: int,
                          stride: int,
                          truncation: bool,
                          add_special_tokens: bool,
                          column: str) -> MultiInferenceNLPMessage:
        """
        For NLP category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows received from Deserialized stage.
        vocab_hashfile : str
            Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
            using the `cudf.utils.hash_vocab_utils.hash_vocab` function
        do_lower_case : bool
            If set to true, original text will be lowercased before encoding.
        seq_len : int
            Limits the length of the sequence returned. If tokenized string is shorter than max_length, output will be
            padded with 0s. If the tokenized string is longer than max_length and do_truncate == False, there will be
            multiple returned sequences containing the overflowing token-ids.
        stride : int
            If do_truncate == False and the tokenized string is larger than max_length, the sequences containing the
            overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to
            stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be
            repeated on the second sequence and so on until the entire sentence is encoded.
        truncation : bool
            If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly
            one output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
            than generated tokens.
        column : str
            Name of the column containing the data that needs to be preprocessed.

        Returns
        -------
        morpheus.messages.MultiInferenceNLPMessage
            infer_message

        """

        text_ser = cudf.Series(x.get_meta(column))

        for symbol in string.punctuation:
            text_ser = text_ser.str.replace(symbol, ' ' + symbol + ' ')

        tokenized = tokenize_text_series(vocab_hash_file=vocab_hash_file,
                                         do_lower_case=do_lower_case,
                                         text_ser=text_ser,
                                         seq_len=seq_len,
                                         stride=stride,
                                         truncation=truncation,
                                         add_special_tokens=add_special_tokens)
        del text_ser

        seg_ids = tokenized.segment_ids
        seg_ids[:, 0] = seg_ids[:, 0] + x.mess_offset

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryNLP(count=tokenized.input_ids.shape[0],
                                    input_ids=tokenized.input_ids,
                                    input_mask=tokenized.input_mask,
                                    seq_ids=seg_ids)

        infer_message = MultiInferenceNLPMessage.from_message(x, memory=memory)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:

        return partial(PreprocessLogParsingStage.pre_process_batch,
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens,
                       column=self._column)

    def _get_preprocess_node(self, builder: mrc.Builder):
        return _stages.PreprocessNLPStage(builder,
                                          self.unique_name,
                                          self._vocab_hash_file,
                                          self._seq_length,
                                          self._truncation,
                                          self._do_lower_case,
                                          self._add_special_tokens,
                                          self._stride,
                                          self._column)
