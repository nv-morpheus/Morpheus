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
import tempfile
import time
from curses import wrapper

import gitutils  # noqa: E402

LFS_DATASETS = {
    'all': '**',
    'examples': 'examples/**',
    'models': 'models/**',
    'tests': 'tests/**',
    'validation': 'models/datasets/validation-data/**'
}


def lfsPull(stdscr, include_paths, poll_interval=1):
    """
    Performs a git lfs pull.
    This can take upwards of a minute to complete we will make use of the
    GIT_LFS_PROGRESS hook to send progress to a file which we can then tail
    while the command is executing.
    """
    cmd = 'git lfs pull -I "{}"'.format(','.join(include_paths))

    with tempfile.NamedTemporaryFile() as progress_file:
        env = os.environ.copy()
        env['GIT_LFS_PROGRESS'] = progress_file.name
        popen = subprocess.Popen(cmd,
                                 env=env,
                                 shell=True,
                                 universal_newlines=True,
                                 stderr=subprocess.STDOUT,
                                 stdout=subprocess.PIPE)

        returncode = None
        while returncode is None:
            time.sleep(poll_interval)
            progress = progress_file.read().decode("UTF-8")
            if progress != '':
                stdscr.addstr(progress)

            returncode = popen.poll()

        # Check if we have any output
        output = popen.stdout.read().rstrip("\n")

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode=returncode, cmd=cmd, output=output)

        logging.info('')
        if output != '':
            logging.info(output)

        return output


def parse_args():
    argparser = argparse.ArgumentParser("Fetches data not included in the repository by default")
    argparser.add_argument("data_set",
                           nargs='*',
                           choices=LFS_DATASETS.keys(),
                           help="Data set to fetch")
    args = argparser.parse_args()
    return args


def main(stdscr):
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    include_paths = [LFS_DATASETS[p] for p in args.data_set]

    lfsPull(stdscr, include_paths)


if __name__ == "__main__":
    wrapper(main)
