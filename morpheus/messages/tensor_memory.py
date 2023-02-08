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


@dataclasses.dataclass(init=False)
class TensorMemory(MessageData, cpp_class=_messages.TensorMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in cupy arrays.

    Parameters
    ----------
    count : int
        Length of each tensor contained in `tensors`.
    tensors : typing.Dict[str, cupy.ndarray]
        Collection of tensors uniquely identified by a name.

    """
    count: int

    tensors: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, repr=False)

    def __init__(self, count: int, tensors: typing.Dict[str, cp.ndarray] = {}):
        self.count = count
        self._tensors = tensors

    def get_tensors(self):
        """
        Get the tensors contained by this instance. It is important to note that when C++ execution is enabled the
        returned tensors will be a Python copy of the tensors stored in the C++ object. As such any changes made to the
        tensors will need to be updated with a call to `set_tensors`.

        Returns
        -------
        typing.Dict[str, cp.ndarray]
        """
        return self._tensors

    def set_tensors(self, tensors):
        """
        Overwrite the tensors stored by this instance. If the length of the tensors has changed, then the `count`
        properte should also be updated.

        Parameters
        ----------
        tensors : typing.Dict[str, cupy.ndarray]
            Collection of tensors uniquely identified by a name.
        """
        self._tensors = tensors
