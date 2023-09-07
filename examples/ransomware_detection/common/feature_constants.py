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


class FeatureConstants():

    FULL_MEMORY_ADDRESS = 2147483647

    PROTECTIONS = {
        'PAGE_READONLY ': 'PAGE_READONLY_RATIO',
        'PAGE_EXECUTE_WRITECOPY ': 'PAGE_EXECUTE_WRITECOPY_RATIO',
        'PAGE_READWRITE ': 'PAGE_READWRITE_RATIO',
        'PAGE_NOACCESS ': 'PAGE_NOACCESS_RATIO',
        'PAGE_EXECUTE_READWRITE ': 'PAGE_EXECUTE_READWRITE_RATIO'
    }

    STATE_LIST = [2,4,5]
    WAIT_REASON_LIST = [9,13,15,22,31]
