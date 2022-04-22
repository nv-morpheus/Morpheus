"""
Example Usage:
python phish-bert-20220113-inference-script.py --validationdata phishing-email-validation-data.jsonlines --model phishing-bert-202111006.onnx --vocab bert-base-uncased-hash.txt --output phishing-email-validation-output.jsonlines
"""

import cudf
from clx.analytics.binary_sequence_classifier import BinarySequenceClassifier;
import argparse;
import s3fs;
from os import path;
import logging
import json
import torch
import onnxruntime
from torch.utils.dlpack import from_dlpack, to_dlpack
import cupy
from scipy.special import expit
import numpy as np

def infer(validationdata,vocab,model,output):

    MODEL_FILE = model
    max_seq_len = 128


    def bert_uncased_tokenize(strings, max_seq_len):
        """
        converts cudf.Series of strings to two torch tensors- token ids and attention mask with padding
        """
        num_strings = len(strings)
        token_ids, mask = strings.str.subword_tokenize(
            vocab,
            max_length=max_seq_len,
            stride=max_seq_len,
            do_lower=True,
            do_truncate=True,
        )[:2]

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(
            token_ids.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack()
        )
        attention_mask = from_dlpack(
            mask.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack()
        )
        return input_ids.type(torch.long), attention_mask.type(torch.long)

    data = []
    with open(validationdata) as f:
        for line in f:
            data.append(json.loads(line))

    df=cudf.DataFrame(data)
    cudf_input = df["data"]
    input_ids, att_masks = bert_uncased_tokenize(cudf_input, 128)
    # moving inputs to host for test

    input_ids = input_ids.detach().cpu().numpy()
    att_masks = att_masks.detach().cpu().numpy()
    print("Running Inference")
    ort_session = onnxruntime.InferenceSession(MODEL_FILE)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids,
                    ort_session.get_inputs()[1].name: att_masks}
    ort_outs = ort_session.run(None, ort_inputs)

    probs = expit(ort_outs[0])

    preds = (probs >= 0.5).astype(np.int_)

    preds = preds[:, 1].tolist()
    bool_list = list(map(bool,preds))
    df["pred"]=bool_list
    print("writing the output file")
    df.to_json(output,orient='records', lines=True)


def main():

    infer(args.validationdata,args.vocab,args.model,args.output)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validationdata", required=True,help="Labelled data in JSON format")
    parser.add_argument("--vocab", required=True, help="BERT voabulary file")
    parser.add_argument("--model", required=True, help="pretrained model")
    parser.add_argument("--output", required=True, help="output filename")
    args = parser.parse_args()

main()