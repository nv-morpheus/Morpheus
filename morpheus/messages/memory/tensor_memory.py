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

import dataclasses
import typing

import cupy as cp

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageData

TensorMemory = _messages.TensorMemory

def get_tensor_prop(self, name: str):
    """
    This method is intended to be used by propery methods in subclasses

    Parameters
    ----------
    name : str
        Tensor key name.

    Returns
    -------
    cupy.ndarray
        Tensor.

    Raises
    ------
    AttributeError
        If tensor name does not exist in the container.
    """
    try:
        return self.get_tensor(name)
    except KeyError:
        raise AttributeError
