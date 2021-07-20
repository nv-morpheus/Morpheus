# Copyright (c) 2021, NVIDIA CORPORATION.
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

import collections

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

Feature = collections.namedtuple(  # pylint: disable=invalid-name
    "Feature", ["input_ids", "input_mask", "segment_ids"])


def create_vocab_table(vocabpath):
    """
    Create Vocabulary tables from the vocab.txt file

    Parameters
    ----------
    vocabpath : str
        Path of vocablary file

    Returns
    -------
    np.array
        id2vocab: np.array, dtype=<U5

    """
    id2vocab = []
    vocab2id = {}
    import numpy as np
    with open(vocabpath) as f:
        for index, line in enumerate(f):
            token = line.split()[0]
            id2vocab.append(token)
            vocab2id[token] = index
    return np.array(id2vocab), vocab2id


def create_tokenizer(vocab_hash_file: str, do_lower_case: bool):
    tokenizer = SubwordTokenizer(vocab_hash_file, do_lower_case=do_lower_case)

    return tokenizer


def tokenize_text_series(tokenizer: SubwordTokenizer,
                         text_ser: cudf.Series,
                         seq_len: int,
                         stride: int,
                         truncation: bool,
                         add_special_tokens: bool):
    """
    This function tokenizes a text series using the bert subword_tokenizer and vocab-hash

    Parameters
    ----------
    text_ser : cudf.Series
        Text Series to tokenize
    seq_len : int
        Sequence Length to use (We add to special tokens for ner classification job)
    stride : int
        Stride for the tokenizer
    vocab_hash_file : str
        vocab_hash_file to use (Created using `perfect_hash.py` with compact flag)

    Returns
    -------
    dict
        A dictionary with these keys {'token_ar':,'attention_ar':,'metadata':}

    """

    assert tokenizer is not None, "Must create tokenizer first using `create_tokenizer()`"

    if len(text_ser) == 0:
        return Feature(None, None, None)

    max_rows_tensor = len(text_ser) * 2
    max_length = seq_len

    # Call the tokenizer
    tokenizer_output = tokenizer(text_ser,
                                 max_length=max_length,
                                 max_num_rows=max_rows_tensor,
                                 add_special_tokens=add_special_tokens,
                                 padding='max_length',
                                 truncation=truncation,
                                 stride=stride,
                                 return_tensors="cp",
                                 return_token_type_ids=False)

    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    metadata = tokenizer_output["metadata"]

    output_f = Feature(input_ids=input_ids, input_mask=attention_mask, segment_ids=metadata)

    return output_f
