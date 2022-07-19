#!/bin/env python3
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
#

import argparse
import csv
import io


def parse_args():
    argparser = argparse.ArgumentParser("Strips the first column in a csv file sending the results to stdout")
    argparser.add_argument("input_file", help="File to strip the column from")
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    output = io.StringIO()
    with open(args.input_file) as fh:
        reader = csv.reader(fh)
        for row in reader:
            writer = csv.writer(output)
            writer.writerow(row[1:])

    print(output.getvalue().strip())


if __name__ == "__main__":
    main()
