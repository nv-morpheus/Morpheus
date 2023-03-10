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

import numpy as np
import pytest
import torch

from morpheus.models.dfencoder import scalers


@pytest.fixture(scope="function")
def standard_scaler():
    scaler = scalers.StandardScaler()
    tensor = torch.tensor([4.4, 5.3, 6.5])
    scaler.fit(tensor)
    yield scaler


def test_ensure_float_type():
    result = scalers.ensure_float_type(np.ones(10, np.int32))
    assert result.dtype == np.float64

    result = scalers.ensure_float_type(torch.ones(10, dtype=torch.int32))
    assert result.dtype == torch.float32

    with pytest.raises(ValueError):
        scalers.ensure_float_type([1, 2, 3])


def test_standard_scaler_fit(standard_scaler):
    assert round(standard_scaler.mean, 2) == 5.4
    assert round(standard_scaler.std, 2) == 1.05

    # Test corner case where all values are the same
    standard_scaler.fit(torch.ones(5, dtype=torch.float64))
    assert standard_scaler.mean == 1
    assert standard_scaler.std == 1.0


def test_standard_scaler_transform(standard_scaler):
    results = standard_scaler.transform(torch.tensor([7.4, 8.3, 9.5]))
    expected = torch.tensor([1.9, 2.75, 3.89])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_standard_scaler_inverse_transform(standard_scaler):
    results = standard_scaler.inverse_transform(torch.tensor([7.4, 8.3, 9.5]))
    expected = torch.tensor([13.2, 14.14, 15.41])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"


def test_standard_scaler_inverse_transform(standard_scaler):
    results = standard_scaler.fit_transform(torch.tensor([7.4, 8.3, 9.5]))
    expected = torch.tensor([-0.95, -0.09, 1.04])
    assert torch.equal(torch.round(results, decimals=2), expected), f"{results} != {expected}"
