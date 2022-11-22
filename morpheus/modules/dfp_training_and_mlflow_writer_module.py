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
from morpheus.modules.module_factory import ModuleFactory

import srf
from dfencoder import AutoEncoder
from srf.core import operators as ops

from morpheus.modules.module import Module
from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from morpheus.messages.multi_dfp_message import MultiDFPMessage

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTrainingMLFlowWriterModule(Module):

    def __init__(self, config: Config, module_config: typing.Dict = {}, model_kwargs: dict = None):

        super().__init__(config, module_config)

    def register_module(self, unique_name: str) -> Module:
        training_config = self._module_config["DFPTraining"]
        mlfowwriter_config = self._module_config["DFPMLFlowModelWriter"]

        ModuleFactory.register_module(self._config, training_config, unique_name)
        ModuleFactory.register_module(self._config, mlfowwriter_config, unique_name)

        def module_init(builder: srf.Builder):
            training_module = builder.load_module(training_config["module_id"],
                                                  training_config["module_namespace"],
                                                  training_config["module_name"],
                                                  training_config)
            
            mlfowwriter_module = builder.load_module(mlfowwriter_config["module_id"],
                                                  mlfowwriter_config["module_namespace"],
                                                  mlfowwriter_config["module_name"],
                                                  mlfowwriter_config)
            
            builder.make_edge(training_module.output_port("output"), mlfowwriter_module.input("input"))
            builder.register_module_input("input", training_module.input_port("input"))
            builder.register_module_output("output", mlfowwriter_module.output_port("output"))

        self._registry.register_module(self._module_id, self._module_namespace, self._version, module_init)
