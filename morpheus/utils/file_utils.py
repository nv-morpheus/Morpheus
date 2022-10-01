# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing

import morpheus


def get_data_file_path(data_filename: str) -> str:
    """
    Get data file path. Also handles paths relative to Morpheus root.

    Parameters
    ----------
    data_filename : str
        Absolute or relative path of data file.

    Returns
    -------
    str
        Data file path.
    """
    # First check if the path is relative
    if (os.path.isabs(data_filename)):
        # Already absolute, nothing to do
        return data_filename

    # See if the file exists.
    does_exist = os.path.exists(data_filename)

    if (not does_exist):
        # If it doesn't exist, then try to make it relative to the morpheus library root
        morpheus_root = os.path.dirname(morpheus.__file__)

        value_abs_to_root = os.path.join(morpheus_root, data_filename)

        # If the file relative to our package exists, use that instead
        if (os.path.exists(value_abs_to_root)):

            return value_abs_to_root

    return data_filename


def load_labels_file(labels_filename: str) -> typing.List[str]:
    """
    Get list of labels from file.

    Parameters
    ----------
    labels_filename : str
        Labels file path

    Returns
    -------
    typing.List[str]
        List of labels
    """

    with open(labels_filename, "r") as lf:
        return [x.strip() for x in lf.readlines()]
