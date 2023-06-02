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

import morpheus._lib.messages as _messages
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta

MultiInferenceMessage = _messages.MultiInferenceMessage

# @dataclasses.dataclass
# class MultiInferenceMessage(_messages.MultiInferenceMessage):
#     """
#     This is a container class that holds the InferenceMemory container and the metadata of the data contained
#     within it. Builds on top of the `MultiTensorMessage` class to add additional data for inferencing.

#     This class requires two separate memory blocks for a batch. One for the message metadata (i.e., start time,
#     IP address, etc.) and another for the raw inference inputs (i.e., input_ids, seq_ids). Since there can be
#     more inference input requests than messages (This happens when some messages get broken into multiple
#     inference requests) this class stores two different offset and count values. `mess_offset` and
#     `mess_count` refer to the offset and count in the message metadata batch and `offset` and `count` index
#     into the inference batch data.
#     """

#     def __init__(self,
#                  *,
#                  meta: _messages.MessageMeta,
#                  mess_offset: int = 0,
#                  mess_count: int = -1,
#                  memory: _messages.TensorMemory = None,
#                  offset: int = 0,
#                  count: int = -1):

#         super().__init__(meta=meta,
#                          mess_offset=mess_offset,
#                          mess_count=mess_count,
#                          memory=memory,
#                          offset=offset,
#                          count=count)

#     @property
#     def inputs(self):
#         """
#         Get inputs stored in the InferenceMemory container. Alias for `MultiInferenceMessage.tensors`.

#         Returns
#         -------
#         cupy.ndarray
#             Inference inputs.

#         """
#         return self.tensors

#     @property
#     def tensors(self):
#         """
#         Get tensors stored in the TensorMemory container sliced according to `offset` and `count`.

#         Returns
#         -------
#         cupy.ndarray
#             Inference tensors.

#         """
#         tensors = self.memory.get_tensors()
#         return {key: self.get_tensor(key) for key in tensors.keys()}

MultiInferenceNLPMessage = _messages.MultiInferenceNLPMessage

# @dataclasses.dataclass
# class MultiInferenceNLPMessage(_messages.MultiInferenceNLPMessage):
#     """
#     A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
#     proper inputs are set and eases debugging.
#     """

#     # required_tensors: typing.ClassVar[typing.List[str]] = ["input_ids", "input_mask", "seq_ids"]

#     def __init__(self,
#                  *,
#                  meta: MessageMeta,
#                  mess_offset: int = 0,
#                  mess_count: int = -1,
#                  memory: TensorMemory = None,
#                  offset: int = 0,
#                  count: int = -1):

#         super().__init__(meta=meta,
#                          mess_offset=mess_offset,
#                          mess_count=mess_count,
#                          memory=memory,
#                          offset=offset,
#                          count=count)

MultiInferenceFILMessage = _messages.MultiInferenceFILMessage

# @dataclasses.dataclass
# class MultiInferenceFILMessage(_messages.MultiInferenceFILMessage):
#     """
#     A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
#     proper inputs are set and eases debugging.
#     """

#     # required_tensors: typing.ClassVar[typing.List[str]] = ["input__0", "seq_ids"]

#     def __init__(self,
#                  *,
#                  meta: MessageMeta,
#                  mess_offset: int = 0,
#                  mess_count: int = -1,
#                  memory: TensorMemory = None,
#                  offset: int = 0,
#                  count: int = -1):

#         super().__init__(meta=meta,
#                          mess_offset=mess_offset,
#                          mess_count=mess_count,
#                          memory=memory,
#                          offset=offset,
#                          count=count)
