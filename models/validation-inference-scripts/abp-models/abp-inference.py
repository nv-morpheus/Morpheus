# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
python abp-inference.py --validationdata abp-validation-data.jsonlines --model abp-nvsmi-xgb-20210310.bst --output abp-validation-output.jsonlines
"""


import xgboost as xgb
import cudf
import json
from cuml import ForestInference
import sklearn.datasets
import cupy
from sklearn.metrics import accuracy_score
import argparse


def infer(validationdata,model,output):
    
    data = []
    with open(validationdata) as f:
        for line in f:
            data.append(json.loads(line))


    df = cudf.DataFrame(data)


    df2=df.drop(['nvidia_smi_log.timestamp','mining'],axis=1)

    # Load the classifier previously saved with xgboost model_save()
    model_path = model

    fm = ForestInference.load(model_path, output_class=True)

    fil_preds_gpu = fm.predict(df2.astype("float32"))

    y_pred = fil_preds_gpu.to_array()

    df2['mining']=y_pred.astype(bool)

    df2.insert(0,'nvidia_smi_log.timestamp',df['nvidia_smi_log.timestamp'])


    df2.to_json(output,orient='records', lines=True)






def main():

    infer(args.validationdata,args.model,args.output)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validationdata", required=True,help="Labelled data in JSON format")
    parser.add_argument("--model", required=True, help="trained model")
    parser.add_argument("--output", required=True, help="output filename")
    args = parser.parse_args()

main()

