# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Message classes, which contain data being transfered between pipeline stages
"""

# Import order is very important here. Import base classes before child ones
# isort: off

from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.memory.inference_memory import InferenceMemoryAE
from morpheus.messages.memory.inference_memory import InferenceMemoryFIL
from morpheus.messages.memory.inference_memory import InferenceMemoryNLP
from morpheus.messages.memory.response_memory import ResponseMemory
from morpheus.messages.memory.response_memory import ResponseMemoryAE
from morpheus.messages.memory.response_memory import ResponseMemoryProbs
from morpheus.messages.message_base import MessageBase
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.messages.multi_inference_message import MultiInferenceFILMessage
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_inference_message import MultiInferenceNLPMessage
from morpheus.messages.multi_response_message import MultiResponseAEMessage
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.messages.multi_response_message import MultiResponseProbsMessage

__all__ = [
    "InferenceMemory",
    "InferenceMemoryAE",
    "InferenceMemoryFIL",
    "InferenceMemoryNLP",
    "MessageBase",
    "MessageMeta",
    "MultiAEMessage",
    "MultiInferenceFILMessage",
    "MultiInferenceMessage",
    "MultiInferenceNLPMessage",
    "MultiMessage",
    "MultiResponseAEMessage",
    "MultiResponseMessage",
    "MultiResponseProbsMessage",
    "ResponseMemory",
    "ResponseMemoryAE",
    "ResponseMemoryProbs",
    "UserMessageMeta",
]
