#!/usr/bin/env python
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

import os
import typing
from collections import OrderedDict

import pandas as pd
import pytest
import torch

from morpheus.config import AEFeatureScalar
from morpheus.models.dfencoder import ae_module
from morpheus.models.dfencoder import autoencoder
from morpheus.models.dfencoder import scalers
from morpheus.models.dfencoder.dataframe import EncoderDataFrame
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager

# Only pandas and Python is supported
pytestmark = [pytest.mark.use_pandas, pytest.mark.use_python]

BIN_COLS = ['ts_anomaly']

CAT_COLS = [
    'apiVersion',
    'errorCode',
    'errorMessage',
    'eventName',
    'eventSource',
    'eventTime',
    'sourceIPAddress',
    'userAgent',
    'userIdentityaccessKeyId',
    'userIdentityaccountId',
    'userIdentityarn',
    'userIdentityprincipalId',
    'userIdentitysessionContextsessionIssueruserName',
    'userIdentitytype'
]

NUMERIC_COLS = ['eventID', 'ae_anomaly_score']


@pytest.fixture(scope="function")
def train_ae():
    """
    Construct an AutoEncoder instance with the same values used by `train_ae_stage`
    """
    yield autoencoder.AutoEncoder(encoder_layers=[512, 500],
                                  decoder_layers=[512],
                                  activation='relu',
                                  swap_p=0.2,
                                  lr=0.01,
                                  lr_decay=.99,
                                  batch_size=512,
                                  verbose=False,
                                  optimizer='sgd',
                                  scaler='standard',
                                  min_cats=1,
                                  progress_bar=False)


@pytest.fixture(scope="function")
def train_df(dataset_pandas: DatasetManager) -> typing.Iterator[pd.DataFrame]:
    yield dataset_pandas[os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-role-g-validation-data-input.csv")]


def compare_numeric_features(features, expected_features):
    assert sorted(features.keys()) == sorted(expected_features.keys())
    for (ft, expected_vals) in expected_features.items():
        ae_vals = features[ft]
        assert round(ae_vals['mean'], 2) == expected_vals['mean'], \
            f"Mean value of feature:{ft} does not match {round(ae_vals['mean'], 2)}!= {expected_vals['mean']}"

        assert round(ae_vals['std'], 2) == expected_vals['std'], \
            f"Mean value of feature:{ft} does not match {round(ae_vals['std'], 2)}!= {expected_vals['std']}"

        assert isinstance(ae_vals['scaler'], expected_vals['scaler_cls'])


def test_ohe():
    tensor = torch.tensor(range(4), dtype=torch.int64)
    results = autoencoder._ohe(tensor, 4, device="cpu")
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert results.device.type == "cpu"
    assert torch.equal(results, expected), f"{results} != {expected}"

    results = autoencoder._ohe(tensor.to("cuda", copy=True), 4, device="cuda")
    assert results.device.type == "cuda"
    assert torch.equal(results, expected.to("cuda", copy=True)), f"{results} != {expected}"


def test_compute_embedding_size():
    for (input, expected) in [(0, 0), (5, 4), (20, 9), (40000, 600)]:
        assert ae_module._compute_embedding_size(input) == expected


def test_complete_layer_constructor():
    cc = ae_module.CompleteLayer(4, 5)
    assert len(cc.layers) == 1
    assert isinstance(cc.layers[0], torch.nn.Linear)
    assert cc.layers[0].in_features == 4
    assert cc.layers[0].out_features == 5

    cc = ae_module.CompleteLayer(4, 5, activation='tanh')
    assert len(cc.layers) == 2
    assert cc.layers[1] is torch.tanh

    cc = ae_module.CompleteLayer(4, 5, dropout=0.2)
    assert len(cc.layers) == 2
    assert isinstance(cc.layers[1], torch.nn.Dropout)
    assert cc.layers[1].p == 0.2

    cc = ae_module.CompleteLayer(6, 11, activation='sigmoid', dropout=0.3)
    assert len(cc.layers) == 3
    assert isinstance(cc.layers[0], torch.nn.Linear)
    assert cc.layers[0].in_features == 6
    assert cc.layers[0].out_features == 11
    assert cc.layers[1] is torch.sigmoid
    assert isinstance(cc.layers[2], torch.nn.Dropout)
    assert cc.layers[2].p == 0.3


def test_complete_layer_interpret_activation():
    cc = ae_module.CompleteLayer(4, 5)
    assert cc.interpret_activation('elu') is torch.nn.functional.elu

    # Test for bad activation, this really does raise the base Exception class.
    with pytest.raises(Exception):
        cc.interpret_activation()

    with pytest.raises(Exception):
        cc.interpret_activation("does_not_exist")

    cc = ae_module.CompleteLayer(6, 11, activation='sigmoid')
    cc.interpret_activation() is torch.sigmoid


@pytest.mark.usefixtures("manual_seed")
def test_complete_layer_forward():
    # Setting dropout probability to 0. The results of dropout our deterministic, but are only
    # consistent when run on the same GPU.
    cc = ae_module.CompleteLayer(3, 5, activation='tanh', dropout=0)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
    results = cc.forward(t)
    expected = torch.tensor([[0.7223, 0.7902, 0.9647, 0.5613, 0.9163], [0.9971, 0.9897, 0.9988, 0.8317, 0.9992],
                             [1.0000, 0.9995, 1.0000, 0.9417, 1.0000], [1.0000, 1.0000, 1.0000, 0.9806, 1.0000]],
                            dtype=torch.float32)

    assert torch.equal(torch.round(results, decimals=4), expected), f"{results} != {expected}"


def test_auto_encoder_constructor_default_vals():
    ae = autoencoder.AutoEncoder()
    assert isinstance(ae.model, torch.nn.Module)
    assert ae.model.encoder_layers is None
    assert ae.model.decoder_layers is None
    assert ae.min_cats == 10
    assert ae.swap_p == 0.15
    assert ae.batch_size == 256
    assert ae.eval_batch_size == 1024
    assert ae.model.activation == 'relu'
    assert ae.optimizer == 'adam'
    assert ae.lr == 0.01
    assert ae.lr_decay is None
    assert ae.device.type == 'cuda'
    assert ae.scaler == 'standard'
    assert ae.loss_scaler is scalers.StandardScaler
    assert ae.n_megabatches == 1


def test_auto_encoder_constructor(train_ae: autoencoder.AutoEncoder):
    """
    Test copnstructor invokation using the values used by `train_ae_stage`
    """
    assert isinstance(train_ae.model, torch.nn.Module)
    assert train_ae.model.encoder_layers == [512, 500]
    assert train_ae.model.decoder_layers == [512]
    assert train_ae.min_cats == 1
    assert train_ae.swap_p == 0.2
    assert train_ae.batch_size == 512
    assert train_ae.eval_batch_size == 1024
    assert train_ae.model.activation == 'relu'
    assert train_ae.optimizer == 'sgd'
    assert train_ae.lr == 0.01
    assert train_ae.lr_decay == 0.99
    assert not train_ae.progress_bar
    assert not train_ae.verbose
    assert train_ae.device.type == 'cuda'
    assert train_ae.scaler == 'standard'
    assert train_ae.loss_scaler is scalers.StandardScaler
    assert train_ae.n_megabatches == 1


def test_auto_encoder_get_scaler():
    ae = autoencoder.AutoEncoder()

    # Test the values in the `AEFeatureScalar` enum
    test_values = [(AEFeatureScalar.NONE, scalers.NullScaler), (AEFeatureScalar.STANDARD, scalers.StandardScaler),
                   (AEFeatureScalar.GAUSSRANK, scalers.GaussRankScaler)]

    assert len(test_values) == len(AEFeatureScalar), "Not all values in AEFeatureScalar are tested"

    for (value, expected) in test_values:
        assert ae.get_scaler(value) is expected


def test_auto_encoder_init_numeric(filter_probs_df):
    ae = autoencoder.AutoEncoder()
    ae._init_numeric(filter_probs_df)

    expected_features = {
        'v1': {
            'mean': 0.46, 'std': 0.35, 'scaler_cls': scalers.StandardScaler
        },
        'v2': {
            'mean': 0.51, 'std': 0.31, 'scaler_cls': scalers.StandardScaler
        },
        'v3': {
            'mean': 0.46, 'std': 0.3, 'scaler_cls': scalers.StandardScaler
        },
        'v4': {
            'mean': 0.54, 'std': 0.27, 'scaler_cls': scalers.StandardScaler
        }
    }

    # AE stores the features in an OrderedDict, but we don't want to be dependent on the order that Pandas reads in the
    # columns of a dataframe.
    assert sorted(ae.num_names) == sorted(expected_features.keys())
    compare_numeric_features(ae.numeric_fts, expected_features)


def test_auto_encoder_fit(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)

    expected_numeric_features = {
        'eventID': {
            'mean': 156.5, 'std': 90.79, 'scaler_cls': scalers.StandardScaler
        },
        'ae_anomaly_score': {
            'mean': 1.67, 'std': 0.38, 'scaler_cls': scalers.StandardScaler
        }
    }

    assert sorted(train_ae.num_names) == sorted(NUMERIC_COLS)
    compare_numeric_features(train_ae.numeric_fts, expected_numeric_features)

    expected_bin_features = {'ts_anomaly': {'cats': [True, False], True: True, False: False}}
    assert train_ae.bin_names == BIN_COLS
    assert train_ae.binary_fts == expected_bin_features

    assert sorted(train_ae.categorical_fts.keys()) == CAT_COLS
    for cat in CAT_COLS:
        assert sorted(train_ae.categorical_fts[cat]['cats']) == sorted(train_df[cat].dropna().unique())

    assert len(train_ae.cyclical_fts) == 0

    all_feature_names = sorted(NUMERIC_COLS + BIN_COLS + CAT_COLS)

    assert sorted(train_ae.feature_loss_stats.keys()) == all_feature_names
    for ft in train_ae.feature_loss_stats.values():
        assert isinstance(ft['scaler'], scalers.StandardScaler)

    assert isinstance(train_ae.optim, torch.optim.SGD)
    assert isinstance(train_ae.lr_decay, torch.optim.lr_scheduler.ExponentialLR)
    assert train_ae.lr_decay.gamma == 0.99
    train_ae.optim is train_ae.lr_decay.optimizer


@pytest.mark.usefixtures("manual_seed")
def test_auto_encoder_get_anomaly_score(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)
    anomaly_score = train_ae.get_anomaly_score(train_df)
    assert len(anomaly_score) == len(train_df)
    assert round(anomaly_score.mean().item(), 2) == 2.28
    assert round(anomaly_score.std().item(), 2) == 0.11


def test_auto_encoder_get_anomaly_score_losses(train_ae: autoencoder.AutoEncoder):
    # create a dummy DataFrame with numerical and boolean features only
    row_cnt = 10
    data = {
        'num_1': [i for i in range(row_cnt)], 
        'bool_1': [i%2 == 0 for i in range(row_cnt)], 
        'bool_2': [i%3 == 0 for i in range(row_cnt)]
    }
    df = pd.DataFrame(data)

    train_ae._build_model(df)

    # call the function and check the output
    mse_loss, bce_loss, cce_loss = train_ae.get_anomaly_score_losses(df)

    # check that the output is of the correct shape
    assert mse_loss.shape == torch.Size([row_cnt, 1]), "mse_loss has incorrect shape"
    assert bce_loss.shape == torch.Size([row_cnt, 2]), "bce_loss has incorrect shape"
    assert cce_loss.shape == torch.Size([row_cnt, 0]), "cce_loss has incorrect shape"


    # create a dummy DataFrame with categorical features
    data = {
        'num_1': [i for i in range(row_cnt)], 
        'num_2': [i/2 for i in range(row_cnt)], 
        'num_3': [i/2 for i in range(row_cnt)], 
        'bool_1': [i%2 == 0 for i in range(row_cnt)], 
        'bool_2': [i%3 == 0 for i in range(row_cnt)],
        'cat_1': [f'str_{i}' for i in range(row_cnt)]
    }
    df = pd.DataFrame(data)
    
    # reset the model
    train_ae.bin_names = []
    train_ae.binary_fts = OrderedDict()
    train_ae.num_names = []
    train_ae.numeric_fts = OrderedDict()
    train_ae._build_model(df)

    # call the function and check the output
    mse_loss, bce_loss, cce_loss = train_ae.get_anomaly_score_losses(df)

    # check that the output is of the correct shape
    assert mse_loss.shape == torch.Size([row_cnt, 3]), "mse_loss has incorrect shape"
    assert bce_loss.shape == torch.Size([row_cnt, 2]), "bce_loss has incorrect shape"
    assert cce_loss.shape == torch.Size([row_cnt, 1]), "cce_loss has incorrect shape"

def test_auto_encoder_prepare_df(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)

    dfc = train_df.copy(deep=True)
    prepared_df = train_ae.prepare_df(dfc)

    assert isinstance(prepared_df, EncoderDataFrame)

    for (i, ft) in enumerate(NUMERIC_COLS):
        scaler = scalers.StandardScaler()
        scaler.fit(train_df[ft].values)
        expected_values = scaler.transform(train_df[ft].values.copy())

        assert (prepared_df[ft].values == expected_values).all(), \
            f"Values for feature {ft} do not match {prepared_df[ft]} != {expected_values}"

    # Bin features should remain the same when the input is already boolean, this DF only has one
    assert (prepared_df.ts_anomaly == train_df.ts_anomaly).all()

    for cat in CAT_COLS:
        assert cat in prepared_df
        isinstance(prepared_df[cat], pd.Categorical)
        assert not prepared_df[cat].hasnans

        if train_df[cat].hasnans:
            assert '_other' in prepared_df[cat].values
        else:
            assert '_other' not in prepared_df[cat].values


def test_build_input_tensor(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)
    prepared_df = train_ae.prepare_df(train_df)
    tensor = train_ae.build_input_tensor(prepared_df)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == 'cuda'
    assert len(tensor) == len(train_df)


@pytest.mark.usefixtures("manual_seed")
def test_auto_encoder_get_results(train_ae: autoencoder.AutoEncoder, train_df: pd.DataFrame):
    train_ae.fit(train_df, epochs=1)
    results = train_ae.get_results(train_df)

    for ft in sorted(NUMERIC_COLS + BIN_COLS + CAT_COLS):
        assert ft in results.columns
        assert f'{ft}_pred' in results.columns
        assert f'{ft}_loss' in results.columns
        assert f'{ft}_z_loss' in results.columns

    assert 'max_abs_z' in results.columns
    assert 'mean_abs_z' in results.columns

    assert round(results.loc[0, 'max_abs_z'], 2) == 2.5

    # Not sure why but numpy.float32(0.33) != 0.33
    assert round(float(results.loc[0, 'mean_abs_z']), 2) == 0.33
    assert results.loc[0, 'z_loss_scaler_type'] == 'z'
