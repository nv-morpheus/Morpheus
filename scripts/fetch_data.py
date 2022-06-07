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
import os
import subprocess
import time

LFS_DATASETS = {
    'all': '**',
    'examples': 'examples/**',
    'models': 'models/**',
    'tests': 'tests/**',
    'validation': 'models/datasets/validation-data/**'
}


def lfsPull(include_paths, poll_interval=0.1):
    """
    Performs a git lfs pull.
    """
    cmd = 'git lfs pull -I "{}"'.format(','.join(include_paths))
    env = os.environ.copy()

    # Instruct git lfs to not supress progress output. Fetching the models can
    # take over a minute to complete, so we want our users to receive feedback.
    env['GIT_LFS_FORCE_PROGRESS'] = '1'
    popen = subprocess.Popen(cmd,
                                env=env,
                                shell=True,
                                universal_newlines=True,
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)

    outpipe = popen.stdout
    returncode = None
    all_out = []
    while returncode is None:
        time.sleep(poll_interval)
        out = outpipe.readline()
        if out.rstrip() != '':
            logging.info(out.rstrip())
            all_out.append(out)

        returncode = popen.poll()

    # Check if we have any additional output written to the pipe before our last poll
    out = outpipe.read()
    if out != '':
        all_out.append(out)

    output = ''.join(all_out).rstrip("\n")
    if returncode != 0:
        logging.error(output)
        raise subprocess.CalledProcessError(returncode=returncode, cmd=cmd, output=output)

    return output


def parse_args():
    argparser = argparse.ArgumentParser("Fetches data not included in the repository by default")
    argparser.add_argument("data_set",
                           nargs='*',
                           choices=list(LFS_DATASETS.keys()),
                           help="Data set to fetch")
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    include_paths = [LFS_DATASETS[p] for p in args.data_set]

    lfsPull(include_paths)


if __name__ == "__main__":
    main()
