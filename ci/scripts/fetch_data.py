#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import logging

import gitutils  # noqa: E402

LFS_DATASETS = {
    'all': '**',
    'examples': 'examples/**',
    'models': 'models/**',
    'tests': 'tests/**',
    'validation': 'models/datasets/validation-data/**'
}


def parse_args():
    argparser = argparse.ArgumentParser("Fetches data not included in the repository by default")
    argparser.add_argument("data_set",
                           nargs='*',
                           choices=LFS_DATASETS.keys(),
                           help="Data set to fetch")
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    include_paths = [LFS_DATASETS[p] for p in args.data_set]

    print(gitutils.lfsPull(include_paths))


if __name__ == "__main__":
    main()
