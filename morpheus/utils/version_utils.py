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

import typing

import srf


def get_srf_version_as_list() -> typing.List[int]:
    """
    This function returns the SRF version as a list.

    Returns
    -------
    typing.List
        ver_list : typing.List
    """
    ver_list = srf.__version__.split('.')
    ver_list = [int(i) for i in ver_list]
    return ver_list
