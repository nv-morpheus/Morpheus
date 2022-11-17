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
python abp-nvsmi-xgb-20210310.py \
       --trainingdata \
       ../../datasets/training-data/abp-sample-nvsmi-training-data.json \
"""

import argparse

import xgboost as xgb

import cudf
from sklearn.model_selection import train_test_split


def preprocess(trainingdata):

    # read json data

    df = cudf.read_json(trainingdata)

    # print list of columns

    print(list(df))

    # print labels

    print(df['label'].unique())

    return df


def train_val_split(df):

    (X_train, X_test, y_train, y_test) = \
        train_test_split(df.drop(['label', 'nvidia_smi_log.timestamp'],
                         axis=1), df['label'], train_size=0.8,
                         random_state=1)

    return (X_train, X_test, y_train, y_test)


def train(X_train, X_test, y_train, y_test):

    # move to Dmatrix

    dmatrix_train = xgb.DMatrix(X_train, label=y_train)
    dmatrix_validation = xgb.DMatrix(X_test, label=y_test)

    # Set parameters

    params = {
        'tree_method': 'gpu_hist',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1,
    }
    evallist = [(dmatrix_validation, 'validation'), (dmatrix_train, 'train')]
    num_round = 5

    # Train the model

    bst = xgb.train(params, dmatrix_train, num_round, evallist)
    return bst


def save_model(model):
    model.save_model('./abp-nvsmi-xgb-20210310.bst')


# def eval(model,X_test, y_test):

#     fm = ForestInference.load(model, output_class=True)
#     fil_preds_gpu = fm.predict(X_test.astype("float32"))

#     y_pred = fil_preds_gpu.to_array()
#     y_true = y_test.to_array()

#     acc = accuracy_score(y_true, y_pred)
#     print("Validation_score: ", acc)


def main():
    print('Preprocessing...')
    (X_train, X_test, y_train, y_test) = \
        train_val_split(preprocess(args.trainingdata))
    print('Model Training...')
    model = train(X_train, X_test, y_train, y_test)
    print('Saving Model')
    save_model(model)


#     print("Model Evaluation...")
#     eval(model,X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainingdata', required=True, help='Labelled data in JSON format')
    args = parser.parse_args()

    main()
