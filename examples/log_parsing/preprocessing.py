# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import string
import typing
from functools import partial

import neo

import cudf

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.pipeline.messages import InferenceMemoryNLP
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiInferenceNLPMessage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.preprocessing import PreprocessBaseStage
from morpheus.utils.cudf_subword_helper import tokenize_text_series


class PreprocessLogParsingStage(PreprocessBaseStage):
    """
    NLP usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    vocab_hashfile : str
        Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
        using the `cudf.utils.hash_vocab_utils.hash_vocab` function
    truncation : bool
        If set to true, strings will be truncated and padded to max_length. Each input string will result in exactly one
        output sequence. If set to false, there may be multiple output sequences when the max_length is smaller
        than generated tokens.
    do_lower_case : bool
        If set to true, original text will be lowercased before encoding.
    stride : int
        If `truncation` == False and the tokenized string is larger than max_length, the sequences containing the
        overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to stride
        there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be repeated on
        the second sequence and so on until the entire sentence is encoded.

    """

    def __init__(self,
                 c: Config,
                 vocab_hash_file: str,
                 truncation: bool,
                 do_lower_case: bool,
                 add_special_tokens: bool,
                 stride: int = -1):
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

        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens


    @property
    def name(self) -> str:
        return "preprocess-logparsing"

    @staticmethod
    def pre_process_batch(x: MultiMessage,
                          vocab_hash_file: str,
                          do_lower_case: bool,
                          seq_len: int,
                          stride: int,
                          truncation: bool,
                          add_special_tokens: bool) -> MultiInferenceNLPMessage:
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

        Returns
        -------
        morpheus.messages.MultiInferenceNLPMessage
            infer_message

        """

        text_ser = cudf.Series(x.get_meta("raw"))

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

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryNLP(count=tokenized.input_ids.shape[0],
                                    input_ids=tokenized.input_ids,
                                    input_mask=tokenized.input_mask,
                                    seq_ids=tokenized.segment_ids)

        infer_message = MultiInferenceNLPMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:

        return partial(PreprocessLogParsingStage.pre_process_batch,
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens)

    def _get_preprocess_node(self, seg: neo.Segment):
        return neos.PreprocessNLPStage(seg,
                                       self.unique_name,
                                       self._vocab_hash_file,
                                       self._seq_length,
                                       self._truncation,
                                       self._do_lower_case,
                                       self._add_special_tokens,
                                       self._stride)
