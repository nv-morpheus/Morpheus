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
python phish-bert-training-script.py
"""
import os.path
import zipfile

import binary_sequence_classifier
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import cudf


def preprocessing():
    """
    fetch data and prepare for model training
    """

    # download and unzip data
    if not os.path.isfile("smsspamcollection.zip"):
        URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        response = requests.get(URL)
        open("smsspamcollection.zip", "wb").write(response.content)
        with zipfile.ZipFile("smsspamcollection.zip") as item:
            item.extractall()
    # read into cudf
    df = cudf.read_csv("SMSSpamCollection", delimiter='\t', header=None, names=['spam/ham', 'message'])
    # convert label to binary 0 = ham, 1 = spam
    df["label"] = df["spam/ham"].str.match('spam').astype(int)
    # split into 80% training, 20% testing datasets
    X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], train_size=0.8, random_state=42)

    return (X_train, y_train, X_test, y_test)


def main():

    print("Preprocessing...")
    X_train, y_train, X_test, y_test = preprocessing()

    print("Model Loading...")
    seq_classifier = binary_sequence_classifier.BinarySequenceClassifier()
    seq_classifier.init_model("bert-base-uncased")

    print("Model Training...")
    seq_classifier.train_model(X_train, y_train, epochs=2)

    print("Saving Model")
    seq_classifier.save_model("./phish-bert-model")

    print("Model Evaluation")
    print("Accuracy:")
    print(seq_classifier.evaluate_model(X_test, y_test))
    test_preds = seq_classifier.predict(X_test, batch_size=128)[0].to_numpy()
    true_labels = y_test.to_numpy()
    print("F1 Score:")
    print(f1_score(true_labels, test_preds))


main()
