# Copyright (c) 2023, NVIDIA CORPORATION.
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
import os

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import configure_logging
from .my_compound_module_consumer_stage import MyCompoundOpModuleWrapper
from .my_test_module_consumer_stage import MyPassthroughModuleWrapper

# pylint: disable=unused-import
from . import my_test_module  # noqa: F401
from . import my_test_module_consumer  # noqa: F401
from . import my_test_compound_module  # noqa: F401
# pylint: enable=unused-import


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email_with_addresses.jsonlines')
    config = Config()  # Morpheus config
    module_config = {
        "module_id": "my_compound_module",
        "module_namespace": "my_module_namespace",
        "module_instance_name": "module_instance_name",  # ... other module config params...
    }

    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipeline.add_stage(LinearModulesStage(config, module_config, input_port_name="input_0",
                                          output_port_name="output_0"))
    pipeline.add_stage(MyPassthroughModuleWrapper(config))
    pipeline.add_stage(MyCompoundOpModuleWrapper(config))
    pipeline.add_stage(MonitorStage(config))

    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
