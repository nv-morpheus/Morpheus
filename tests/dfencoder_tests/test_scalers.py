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


def test_ensure_float_type():
    result = scalers.ensure_float_type(np.ones(10, np.int32))
    assert result.dtype == np.float64

    result = scalers.ensure_float_type(torch.ones(10, dtype=torch.int32))
    assert result.dtype == torch.float32

    with pytest.raises(ValueError):
        scalers.ensure_float_type([1, 2, 3])
