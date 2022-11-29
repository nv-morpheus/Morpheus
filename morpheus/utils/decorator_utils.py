# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import srf

registry = srf.ModuleRegistry


def is_module_registered(func):

    def wrapper(*args, **kwargs):

        if args is not None and len(args) >= 2:
            module_id = args[0]._module_id
            namespace = args[0]._namespace
        else:
            module_id = kwargs["module_id"]
            namespace = kwargs["namespace"]

        if not registry.contains(module_id, namespace):
            raise Exception("Module {} doesn't exist in the Namespace {}".format(module_id, namespace))

        return func(*args, **kwargs)

    return wrapper
