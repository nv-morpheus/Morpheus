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
import curses
import logging
import os
import subprocess
import sys
import tempfile
import time

LFS_DATASETS = {
    'all': '**',
    'examples': 'examples/**',
    'models': 'models/**',
    'tests': 'tests/**',
    'validation': 'models/datasets/validation-data/**'
}


def print_line(stdscr, max_x, last_print_len, line):
    print_len = min(len(line), max_x)
    if stdscr is not None:
        if print_len < last_print_len:
            stdscr.move(0, 0)
            stdscr.clrtoeol()

        stdscr.addstr(0, 0, line, print_len)
        stdscr.refresh()
    else:
        logging.info(line)

    return print_len


def lfsPull(stdscr, include_paths, poll_interval=0.1):
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

        if stdscr is not None:
            (_, max_x) = stdscr.getmaxyx()
        else:
            max_x = sys.maxsize

        last_print_len = 0

        returncode = None
        while returncode is None:
            time.sleep(poll_interval)
            progress_lines = progress_file.readlines()
            if len(progress_lines):
                line = progress_lines[-1].decode("UTF8")
                last_print_len = print_line(stdscr, max_x, last_print_len, line)

            returncode = popen.poll()

        # Check if we have any output
        output = popen.stdout.read().rstrip("\n")

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode=returncode, cmd=cmd, output=output)

        logging.info('')
        if output != '':
            logging.info(output)
        else:
            logging.info("Done.")

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

    if os.environ.get('TERM') is not None:
        curses.wrapper(lfsPull, include_paths)
    else:
        lfsPull(None, include_paths)


if __name__ == "__main__":
    main()
