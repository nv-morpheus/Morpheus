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
python root-cause-bert.py \
--trainingdata ../../datasets/training-data/root-cause-training-data.csv \
--unseenerrors ../../datasets/training-data/root-cause-unseen-errors.csv
"""

import argparse
import cudf
import torch
from binary_sequence_classifier import BinarySequenceClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time


def train(trainingdata, unseenerrors):
    """This function splits the data and appends the new errors
    to test set. After running the training it prints the
    evaluation scores."""

    # setting a random seed

    random_seed = 42

    dflogs = pd.read_csv(trainingdata, header=None, names=['label',
                         'log'])

    dfnewerror = pd.read_csv(unseenerrors, header=None, names=['label',
                             'log'])

    (X_train, X_test, y_train, y_test) = train_test_split(
        dflogs, dflogs.label, random_state=random_seed
    )

    X_train.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)

    X_test = pd.concat([X_test, dfnewerror])

    y_test = pd.concat([y_test, dfnewerror['label']])

    X_test.reset_index(drop=True, inplace=True)

    y_test.reset_index(drop=True, inplace=True)

    dfnewerror = cudf.DataFrame.from_pandas(dfnewerror)

    dflogs = cudf.DataFrame.from_pandas(dflogs)

    seq_classifier = BinarySequenceClassifier()

    seq_classifier.init_model('bert-base-uncased')

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    seq_classifier.train_model(X_train['log'], y_train, batch_size=128,
                               epochs=1, learning_rate=2e-04)

    timestr = time.strftime('%Y%m%d-%H%M%S')

    seq_classifier.save_model(timestr)

    print(seq_classifier.evaluate_model(X_test['log'], y_test))

    test_preds = seq_classifier.predict(X_test['log'], batch_size=128,
                                        threshold=0.5)

    tests = test_preds[0].to_array()

    true_labels = X_test['label']

    print(f1_score(true_labels, tests))


def main():

    train(args.trainingdata, args.unseenerrors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainingdata', required=True,
                        help='Labelled data in CSV format')
    parser.add_argument('--unseenerrors', required=True,
                        help="""Labelled data to be added to test set for
                        evaluation after training"""
                        )
    args = parser.parse_args()

main()
