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
class TensorMemory(_messages.TensorMemory, MessageData):
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

    def __init__(self, *, count: int = None, tensors: typing.Dict[str, cp.ndarray] = None):

        if tensors is None:
            tensors = {}

        super().__init__(count=count, tensors=tensors)

        self._check_tensors(tensors)

    def _check_tensors(self, tensors: typing.Dict[str, cp.ndarray]):
        for tensor in tensors.values():
            self._check_tensor(tensor)

    def _check_tensor(self, tensor: cp.ndarray):
        if (tensor.shape[0] != self.count):
            class_name = type(self).__name__
            raise ValueError(
                f"The number rows in tensor {tensor.shape[0]} does not match {class_name}.count of {self.count}")

    def __getattr__(self, name: str) -> typing.Any:
        if ("tensors" in self.__dict__ and self.has_tensor(name)):
            return self.get_tensor(name)

        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)
        raise AttributeError

    def _get_tensor_prop(self, name: str):
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
            return self._tensors[name]
        except KeyError:
            raise AttributeError
