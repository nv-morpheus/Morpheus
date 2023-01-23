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


@dataclasses.dataclass
class TensorMemory(MessageData, cpp_class=_messages.TensorMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in cupy arrays.

    Parameters
    ----------
    count : int
        Number of inference inputs.
    inputs : typing.Dict[str, cupy.ndarray]
        Inference inputs to model.

    """
    count: int

    tensors: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)
