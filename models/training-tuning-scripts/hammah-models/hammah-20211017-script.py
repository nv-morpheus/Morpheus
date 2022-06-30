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
python hammah-20211017-script.py \
       --trainingdata "../datasets/training-data/hammah-user123-training-data.csv"\
       --valdata "../datasets/validation-data/hammah-user123-validation-data.csv" 
"""


import pandas as pd
import os
import numpy as np
import glob
import torch
from dfencoder import AutoEncoder
import matplotlib.pyplot as plt
import argparse
import dill


def main():
    X_train = pd.read_csv(args.trainingdata)
    X_val = pd.read_csv(args.valdata)


    features=['eventSource', 'eventName', 'sourceIPAddress', 'userAgent','userIdentitytype', 'requestParametersroleArn', 'requestParametersroleSessionName','requestParametersdurationSeconds', 'responseElementsassumedRoleUserassumedRoleId','responseElementsassumedRoleUserarn', 'apiVersion', 'userIdentityprincipalId','userIdentityarn', 'userIdentityaccountId', 'userIdentityaccessKeyId','userIdentitysessionContextsessionIssuerprincipalId', 'userIdentitysessionContextsessionIssueruserName','tlsDetailsclientProvidedHostHeader', 'requestParametersownersSetitems','requestParametersmaxResults', 'requestParametersinstancesSetitems','errorCode', 'errorMessage', 'requestParametersmaxItems','responseElementsrequestId', 'responseElementsinstancesSetitems','requestParametersgroupSetitems', 'requestParametersinstanceType','requestParametersmonitoringenabled', 'requestParametersdisableApiTermination','requestParametersebsOptimized', 'responseElementsreservationId', 'requestParametersgroupName'] #NO userIdentitysessionContextsessionIssuerarn,userIdentityuserName
    for i in list(X_train):
        if i not in features:
            X_train = X_train.drop(i, axis=1)
    for i in list(X_val):
        if i not in features:
            X_val = X_val.drop(i, axis=1)




    X_train = X_train.dropna(axis=1, how='all')
    X_val = X_val.dropna(axis=1, how='all')

    for i in list(X_val):
        if i not in list(X_train):
            X_val = X_val.drop([i], axis=1)

    for i in list(X_train):
        if i not in list(X_val):
            X_train = X_train.drop([i], axis=1)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    model = AutoEncoder(
        encoder_layers=[512, 500],  # layers of the encoding part
        decoder_layers=[512],  # layers of the decoding part
        activation='relu',  # activation function
        swap_p=0.2,  # noise parameter
        lr=0.01,  # learning rate
        lr_decay=.99,  # learning decay
        batch_size=512,
        logger='ipynb',
        verbose=False,
        optimizer='sgd',  # SGD optimizer is selected(Stochastic gradient descent)
        scaler='gauss_rank',  # feature scaling method
        min_cats=1  # cut off for minority categories
    )

    model.fit(X_train, epochs=25, val=X_val)

    torch.save(model.state_dict(), args.trainingdata[:-4]+".pkl")
    with open(args.trainingdata[:-4]+'dill'+'.pkl', 'wb') as f:
        serialized_model = dill.dumps(model)
        f.write(serialized_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainingdata", required=True,
                        help="CloudTrail CSV")
    parser.add_argument("--valdata", required=True,
                        help="CloudTrail CSV")
    args = parser.parse_args()

    main()
