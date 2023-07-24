#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import morpheus


def update_link(full_path: str, filename: str):
    """
    Update a broken symlink if a file of the same name exists in the morpheus data dir.
    """
    logging.debug("updating %s: %s", filename, full_path)
    new_link_target = os.path.join(morpheus.DATA_DIR, filename)
    if os.path.exists(new_link_target):
        try:
            os.remove(full_path)
            os.symlink(new_link_target, full_path)
        except Exception as e:
            logging.error("Unexpected error updating symlink: %s", e)
            raise


def main(morpheus_root_dir: str):
    """
    Walk through the models directory and update any broken symlinks if they exist in the morpheus data dir.
    """
    models_dir = os.path.join(morpheus_root_dir, "models")
    for dirpath, _, filenames in os.walk(models_dir, followlinks=False):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if os.path.islink(full_path):
                logging.debug("checking %s: %s", filename, full_path)
                try:
                    os.path.realpath(full_path, strict=True)
                except FileNotFoundError:
                    update_link(full_path, filename)
                except Exception as e:
                    logging.error("Unexpected Error: %s", e)
                    raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Updating symlinks in models directory")
    main(os.environ["MORPHEUS_ROOT"])
