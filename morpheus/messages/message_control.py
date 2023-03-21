# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageBase


class MessageControl(MessageBase, cpp_class=_messages.MessageControl):
    """
    MessageControl is an object that serves as a specification of the tasks to be executed in a pipeline workflow.
    The MessageControl is passed between stages of the pipeline, with each stage executing the tasks specified in
    the MessageControl configuration.

    MessageControl is capable of carrying payload of the MessageMeta type.
    """

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
