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

import importlib
import os

import cupy as cp
import pytest

import cudf

import morpheus._lib.messages as _messages
import morpheus.config
from morpheus import messages
from morpheus.messages.memory import inference_memory
from morpheus.messages.memory import response_memory
from morpheus.messages.memory import tensor_memory


def check_message(python_type: type, cpp_type: type, should_be_cpp: bool, no_cpp_class: bool, *args: tuple, **kwargs):
    instance = python_type(*args, **kwargs)

    # Check that the C++ type is set in the class
    expected_cpp_class = None if no_cpp_class else cpp_type
    assert python_type._cpp_class is expected_cpp_class

    # Check that the isinstance to Python type works
    assert isinstance(instance, python_type)

    # Check that the instantiated class is the right type
    expected_class = cpp_type if should_be_cpp and cpp_type is not None else python_type
    assert instance.__class__ is expected_class


def check_all_messages(should_be_cpp: bool, no_cpp_class: bool):

    df = cudf.DataFrame(range(1), columns="test")

    check_message(messages.MessageMeta, _messages.MessageMeta, should_be_cpp, no_cpp_class, *(df, ))

    # UserMessageMeta doesn't contain a C++ impl, so we should
    # always received the python impl
    check_message(messages.UserMessageMeta, None, should_be_cpp, no_cpp_class, *(None, None))

    check_message(messages.MultiMessage,
                  _messages.MultiMessage,
                  should_be_cpp,
                  no_cpp_class,
                  **{"meta": messages.MessageMeta(df)})

    check_message(tensor_memory.TensorMemory, _messages.TensorMemory, should_be_cpp, no_cpp_class, *(1, ))
    check_message(messages.InferenceMemory, _messages.InferenceMemory, should_be_cpp, no_cpp_class, *(1, ))

    cp_array = cp.zeros((1, 2))

    check_message(messages.InferenceMemoryNLP,
                  _messages.InferenceMemoryNLP,
                  should_be_cpp,
                  no_cpp_class,
                  *(1, cp_array, cp_array, cp_array))

    check_message(messages.InferenceMemoryFIL,
                  _messages.InferenceMemoryFIL,
                  should_be_cpp,
                  no_cpp_class,
                  *(1, cp_array, cp_array))

    # No C++ impl, should always get the Python class
    check_message(messages.InferenceMemoryAE, None, should_be_cpp, no_cpp_class, *(1, cp_array, cp_array))

    multi_tensor_message_tensors = {
        "input_ids": cp.zeros((1, 2)),
        "input_mask": cp.zeros((1, 2)),
        "seq_ids": cp.zeros((1, 3), dtype=cp.int32),
        "input__0": cp.zeros((1, 2)),
        "probs": cp.zeros((1, 2))
    }

    check_message(messages.MultiTensorMessage,
                  _messages.MultiTensorMessage,
                  should_be_cpp,
                  no_cpp_class,
                  meta=messages.MessageMeta(df),
                  memory=tensor_memory.TensorMemory(count=1, tensors=multi_tensor_message_tensors))

    check_message(messages.MultiInferenceMessage,
                  _messages.MultiInferenceMessage,
                  should_be_cpp,
                  no_cpp_class,
                  meta=messages.MessageMeta(df),
                  memory=inference_memory.InferenceMemory(count=1, tensors=multi_tensor_message_tensors))

    check_message(
        messages.MultiInferenceNLPMessage,
        _messages.MultiInferenceNLPMessage,
        should_be_cpp,
        no_cpp_class,
        meta=messages.MessageMeta(df),
        memory=inference_memory.InferenceMemory(count=1, tensors=multi_tensor_message_tensors),
    )

    check_message(messages.MultiInferenceFILMessage,
                  _messages.MultiInferenceFILMessage,
                  should_be_cpp,
                  no_cpp_class,
                  meta=messages.MessageMeta(df),
                  memory=inference_memory.InferenceMemory(count=1, tensors=multi_tensor_message_tensors))

    check_message(messages.ResponseMemory, _messages.ResponseMemory, should_be_cpp, no_cpp_class, *(1, ))

    check_message(messages.ResponseMemoryProbs,
                  _messages.ResponseMemoryProbs,
                  should_be_cpp,
                  no_cpp_class,
                  *(1, cp_array))

    # No C++ impl
    check_message(messages.ResponseMemoryAE, None, should_be_cpp, no_cpp_class, *(1, cp_array))

    check_message(messages.MultiResponseMessage,
                  _messages.MultiResponseMessage,
                  should_be_cpp,
                  no_cpp_class,
                  meta=messages.MessageMeta(df),
                  memory=response_memory.ResponseMemory(count=1, tensors=multi_tensor_message_tensors))

    check_message(messages.MultiResponseProbsMessage,
                  _messages.MultiResponseProbsMessage,
                  should_be_cpp,
                  no_cpp_class,
                  meta=messages.MessageMeta(df),
                  memory=response_memory.ResponseMemoryProbs(count=1, probs=multi_tensor_message_tensors["probs"]))

    # No C++ impl
    check_message(
        messages.MultiResponseAEMessage,
        None,
        should_be_cpp,
        no_cpp_class,
        meta=messages.MessageMeta(df),
        memory=response_memory.ResponseMemoryAE(count=1, probs=multi_tensor_message_tensors["probs"]),
        user_id="",
    )


def test_constructor_cpp(config):
    check_all_messages(morpheus.config.CppConfig.get_should_use_cpp(), False)


@pytest.mark.reload_modules(morpheus.config)
@pytest.mark.usefixtures("reload_modules", "restore_environ")
def test_constructor_env(config):
    # Set the NO_CPP flag which should disable C++ regardless
    os.environ['MORPHEUS_NO_CPP'] = '1'

    # Reload the CppConfig class just in case
    importlib.reload(morpheus.config)

    # Check all messages. Should be False regardless due to the environment variable
    check_all_messages(False, False)
