# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from morpheus._lib.file_types import FileTypes

# def determine_file_type(filename: str) -> FileTypes:
#     # Determine from the file extension
#     ext = os.path.splitext(filename)

#     # Get the extension without the dot
#     ext = ext[1].lower()[1:]

#     # Check against supported options
#     if (ext == "json" or ext == "jsonlines"):
#         return FileTypes.Json
#     elif (ext == "csv"):
#         return FileTypes.Csv
#     else:
#         raise RuntimeError("Unsupported extension '{}' with 'auto' type. 'auto' only works with: csv, json".format(ext))
