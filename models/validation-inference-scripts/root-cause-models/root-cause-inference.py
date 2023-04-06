# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example Usage:
python \
root-cause-inference.py \
    --validationdata ../../datasets/validation-data/root-cause-validation-data-input.jsonlines \
    --model ../../root-cause-models/root-cause-binary-bert-20221118.onnx \
    --vocab ${MORPHEUS_ROOT}/morpheus/data/bert-base-uncased-hash.txt \
    --output root-cause-validation-output.jsonlines
"""

###########################################################################################
# cudf imports moved before torch import to avoid the following error:
# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
###########################################################################################

import argparse
import json

import numpy as np
import onnxruntime
import torch
from scipy.special import expit


def infer(
    validationdata,
    vocab,
    model,
    output,
):

    MODEL_FILE = model

    def bert_uncased_tokenize(strings, max_seq_len):
        """
        converts cudf.Series of strings to two torch tensors and
        meta_data- token ids and attention mask with padding
        """

        tokenizer = SubwordTokenizer(vocab, do_lower_case=True)
        tokenizer_output = tokenizer(
            strings,
            max_length=max_seq_len,
            stride=0,
            truncation=True,
            max_num_rows=len(strings),
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids = tokenizer_output['input_ids'].type(torch.long)
        att_masks = tokenizer_output['attention_mask'].type(torch.long)
        # meta_data = tokenizer_output['metadata']
        del tokenizer_output
        return (input_ids, att_masks)

    data = []
    with open(validationdata) as f:
        for line in f:
            data.append(json.loads(line))

    df = cudf.DataFrame(data)
    cudf_input = df['log']
    (input_ids, att_masks) = bert_uncased_tokenize(cudf_input, 128)

    # moving inputs to host for test

    input_ids = input_ids.detach().cpu().numpy()
    att_masks = att_masks.detach().cpu().numpy()
    print('Running Inference')
    ort_session = onnxruntime.InferenceSession(MODEL_FILE)

    # compute ONNX Runtime output prediction

    ort_inputs = {ort_session.get_inputs()[0].name: input_ids, ort_session.get_inputs()[1].name: att_masks}
    ort_outs = ort_session.run(None, ort_inputs)

    probs = expit(ort_outs[0])

    preds = (probs >= 0.5).astype(np.int_)

    preds = preds[:, 1].tolist()
    bool_list = list(map(bool, preds))
    df['pred'] = bool_list
    print('writing the output file')
    df.to_json(output, orient='records', lines=True)


def main():

    infer(args.validationdata, args.vocab, args.model, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validationdata', required=True, help='Labelled data in JSON format')
    parser.add_argument('--vocab', required=True, help='BERT voabulary file')
    parser.add_argument('--model', required=True, help='pretrained model')
    parser.add_argument('--output', required=True, help='output filename')
    args = parser.parse_args()

main()
