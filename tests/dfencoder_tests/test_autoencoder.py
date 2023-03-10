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

import pytest
import torch

from morpheus.models.dfencoder import autoencoder


def test_ohe():
    tensor = torch.tensor(range(4), dtype=torch.int64)
    results = autoencoder.ohe(tensor, 4, device="cpu")
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert results.device.type == "cpu"
    assert torch.equal(results, expected), f"{results} != {expected}"

    results = autoencoder.ohe(tensor.to("cuda", copy=True), 4, device="cuda")
    assert results.device.type == "cuda"
    assert torch.equal(results, expected.to("cuda", copy=True)), f"{results} != {expected}"
