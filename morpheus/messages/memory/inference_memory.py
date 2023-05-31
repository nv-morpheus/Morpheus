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

import cupy as cp

import morpheus._lib.messages as _messages
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.memory.tensor_memory import TensorMemory


@dataclasses.dataclass(init=False)
class InferenceMemory(_messages.InferenceMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in cupy arrays.
    """
    def __init__(self, *, count: int, tensors: object = None):
        super().__init__(count=count, tensors=tensors)


@dataclasses.dataclass(init=False)
class InferenceMemoryNLP(_messages.InferenceMemoryNLP):
    """
    This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.

    Parameters
    ----------
    input_ids : cupy.ndarray
        The token-ids for each string padded with 0s to max_length.
    input_mask : cupy.ndarray
        The mask for token-ids result where corresponding positions identify valid token-id values.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """

    def __init__(self, *, count: int, input_ids: cp.ndarray, input_mask: cp.ndarray, seq_ids: cp.ndarray):
        super().__init__(count=count, input_ids=input_ids, input_mask=input_mask, seq_ids=seq_ids)


@dataclasses.dataclass(init=False)
class InferenceMemoryFIL(_messages.InferenceMemoryFIL):
    """
    This is a container class for data that needs to be submitted to the inference server for FIL category
    usecases.

    Parameters
    ----------
    input__0 : cupy.ndarray
        Inference input.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input__0: dataclasses.InitVar[cp.ndarray] = DataClassProp(TensorMemory._get_tensor_prop,
                                                              InferenceMemory.set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(TensorMemory._get_tensor_prop,
                                                             InferenceMemory.set_input)

    def __init__(self, *, count: int, input__0: cp.ndarray, seq_ids: cp.ndarray):
        super().__init__(count=count, input__0=input__0, seq_ids=seq_ids)


@dataclasses.dataclass(init=False)
class InferenceMemoryAE(InferenceMemory):
    """
    This is a container class for data that needs to be submitted to the inference server for auto encoder usecases.

    Parameters
    ----------
    input : cupy.ndarray
        Inference input.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).
    """

    input: dataclasses.InitVar[cp.ndarray] = DataClassProp(TensorMemory._get_tensor_prop, InferenceMemory.set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(TensorMemory._get_tensor_prop,
                                                             InferenceMemory.set_input)

    def __init__(self, *, count: int, input: cp.ndarray, seq_ids: cp.ndarray):
        super().__init__(count=count, tensors={'input': input, 'seq_ids': seq_ids})
