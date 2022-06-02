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

import gitutils  # noqa: E402

LFS_DATASETS = ['all', 'examples', 'models', 'tests']

def parse_args():
    argparser = argparse.ArgumentParser("Fetches data not included in the repository by default")
    argparser.add_argument("data_set",
                           nargs='*',
                           choices=LFS_DATASETS,
                           help="Data set to fetch")
    args = argparser.parse_args()
    return args

def main():
    args = parse_args()

    lfs_pull_args = {}
    if 'all' in args.data_set:
        lfs_pull_args['pull_all'] = True
    else:
        lfs_pull_args['include_paths'] = ['{}/**'.format(p) for p in args.data_set]

    print(gitutils.lfsPull(**lfs_pull_args))

if __name__ == "__main__":
    main()
