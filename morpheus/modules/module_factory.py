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

from morpheus.modules.dfp_training_module import DFPTrainingModule
from morpheus.modules.dfp_mlflow_model_writer_module import DFPMLFlowModelWriterModule
from morpheus.modules.dfp_training_and_mlflow_writer_module import DFPTrainingMLFlowWriterModule
from morpheus.modules.abstract_module import AbstractModule

logger = logging.getLogger("morpheus.{}".format(__name__))


class ModuleFactory:

    __cls_dict = {
        "dfp_training": "DFPTrainingModule",
        "dfp_mlflow_model_writer": "DFPMLFlowModelWriterModule",
        "dfp_training_mlflow_model_writer": "DFPTrainingMLFlowWriterModule"
    }

    @staticmethod
    def cls_dict() -> AbstractModule:
        return ModuleFactory.__cls_dict

    class GenerateInstance(object):

        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            class_name, pipeline_config, module_config = self.func(*args, **kwargs)
            try:
                target_cls = globals()[class_name](pipeline_config, module_config)
                return target_cls
            except KeyError as error:
                logger.error(error)
                logger.exception(error)
                raise

    @GenerateInstance
    def get_instance(pipeline_config: typing.Dict, module_config: typing.Dict) -> AbstractModule:
        module = module_config["module_id"].lower()
        if module and module in ModuleFactory.cls_dict():
            return ModuleFactory.cls_dict()[module], pipeline_config, module_config
        else:
            raise KeyError("Module implementation doesn't exists for module { }".format(module))

    @staticmethod
    def register_module(pipeline_config: typing.Dict, module_config: typing.Dict):
        module = ModuleFactory.get_instance(pipeline_config, module_config)
        module.register_module()