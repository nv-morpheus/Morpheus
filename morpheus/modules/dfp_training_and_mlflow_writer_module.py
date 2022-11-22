# Copyright (c) 2022, NVIDIA CORPORATION.
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

import logging
import typing

import srf

from morpheus.modules.abstract_module import AbstractModule
from morpheus.config import Config

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTrainingMLFlowWriterModule(AbstractModule):

    def __init__(self, config: Config, mc: typing.Dict):

        super().__init__(config, mc)

    def on_data():
        pass

    def register_module(self):

        self._register_chained_module()

    
