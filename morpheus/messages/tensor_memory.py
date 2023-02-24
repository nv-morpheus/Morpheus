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

    def __init__(self, count: int, tensors: typing.Dict[str, cp.ndarray] = None):
        self.count = count

        if tensors is None:
            tensors = {}
        else:
            self._check_tensors(tensors)

        self._tensors = tensors

    def _check_tensors(self, tensors: typing.Dict[str, cp.ndarray]):
        for tensor in tensors.values():
            self._check_tensor(tensor)

    def _check_tensor(self, tensor: cp.ndarray):
        if (tensor.shape[0] != self.count):
            class_name = type(self).__name__
            raise ValueError(
                f"The number rows in tensor {tensor.shape[0]} does not match {class_name}.count of {self.count}")

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
        property should also be updated.

        Parameters
        ----------
        tensors : typing.Dict[str, cupy.ndarray]
            Collection of tensors uniquely identified by a name.
        """
        self._check_tensors(tensors)
        self._tensors = tensors

    def get_tensor(self, name):
        """
        Get the Tensor stored in the TensorMemory container identified by `name`.

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
        KeyError
            When no matching tensor exists.
        """
        return self._tensors[name]

    def set_tensor(self, name, tensor):
        """
        Update the tensor identified by `name`. If the length of the tensor has changed, then the `count`
        property should also be updated.

        Parameters
        ----------
        tensors : typing.Dict[str, cupy.ndarray]
            Collection of tensors uniquely identified by a name.
        """
        self._check_tensor(tensor)
        self._tensors[name] = tensor
