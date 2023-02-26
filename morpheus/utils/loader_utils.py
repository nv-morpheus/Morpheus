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

from morpheus._lib.common import DataLoaderRegistry as registry

logger = logging.getLogger(__name__)


def register_loader(loder_id):
    """
    Registers a loader if not exists in the dataloader registry.

    Parameters
    ----------
    loder_id : str
        Unique identifier for a loader in the dataloader registry.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    def inner_func(func):
        # Register a loader if not exists in the registry.
        if not registry.contains(loder_id):
            registry.register_loader(loder_id, func)
            print("Laoder '{}' was successfully registered.".format(loder_id), flush=True)
            logger.debug("Laoder '{}' was successfully registered.".format(loder_id))
        else:
            logger.debug("Module: '{}' already exists.".format(loder_id))
            print("Module: '{}' already exists.".format(loder_id), flush=True)

        return func

    return inner_func
