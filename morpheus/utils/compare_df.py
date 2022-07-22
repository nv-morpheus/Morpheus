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
import re
import sys
import typing

import datacompy
import pandas as pd

from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.logger import configure_logging

logger = logging.getLogger("morpheus.utils.compare_df")


def filter_df(df: pd.DataFrame,
              include_columns: typing.List[str],
              exclude_columns: typing.List[str],
              replace_idx: str = None):

    if (include_columns is not None and len(include_columns) > 0):
        include_columns = re.compile("({})".format("|".join(include_columns)))

    if exclude_columns is not None:
        exclude_columns = [re.compile(x) for x in exclude_columns]
    else:
        exclude_columns = []

    # Filter out any known good/bad columns we dont want to compare
    columns: typing.List[str] = []

    # First build up list of included. If no include regex is specified, select all
    if (include_columns is None or len(include_columns) == 0):
        columns = list(df.columns)
    else:
        columns = [y for y in list(df.columns) if include_columns.match(y)]

    # Now remove by the ignore
    for test in exclude_columns:
        columns = [y for y in columns if not test.match(y)]

    filtered_df = df[columns]

    # if the index column is set, make that the index
    if replace_idx is not None and replace_idx in filtered_df:
        filtered_df = filtered_df.set_index(replace_idx, drop=True)

        if replace_idx.startswith("_index_"):
            filtered_df.index.name = str(filtered_df.index.name).replace("_index_", "", 1)

    return filtered_df


def compare_df(df_a: pd.DataFrame,
               df_b: pd.DataFrame,
               include_columns: typing.List[str] = None,
               exclude_columns: typing.List[str] = None,
               replace_idx: str = None,
               abs_tol: float = 0.001,
               rel_tol: float = 0.005,
               dfa_name: str = "val",
               dfb_name: str = "res"):
    """
    Compares two pandas Dataframe, returning a comparison summary as a dict in the form of:
    ```
    {
        "total_rows": <int>,
        "matching_rows": <int>,
        "diff_rows": <int>,
        "matching_cols": <[str]>,
        "extra_cols": extra_cols: <[str]>,
        "missing_cols": missing_cols: <[str]>,
    }
    ```
    """
    df_a_filtered = filter_df(df_a, include_columns, exclude_columns, replace_idx=replace_idx)
    df_b_filtered = filter_df(df_b, include_columns, exclude_columns, replace_idx=replace_idx)

    missing_columns = df_a_filtered.columns.difference(df_b_filtered.columns)
    extra_columns = df_b_filtered.columns.difference(df_a_filtered.columns)
    same_columns = df_a_filtered.columns.intersection(df_b_filtered.columns)

    # Now get the results in the same order
    df_b_filtered = df_b_filtered[same_columns]

    comparison = datacompy.Compare(
        df_a_filtered,
        df_b_filtered,
        on_index=True,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        df1_name=dfa_name,
        df2_name=dfb_name,
        cast_column_names_lower=False,
    )

    total_rows = len(df_a_filtered)
    diff_rows = len(df_a_filtered) - int(comparison.count_matching_rows())

    if (comparison.matches()):
        logger.info("Results match validation dataset")
    else:
        match_columns = comparison.intersect_rows[same_columns + "_match"]
        mismatched_idx = match_columns[match_columns.apply(lambda r: not r.all(), axis=1)].index

        merged = pd.concat([df_a_filtered, df_b_filtered], keys=[dfa_name, dfb_name]).swaplevel().sort_index()

        mismatch_df = merged.loc[mismatched_idx]

        if diff_rows > 0:
            logger.debug("Results do not match. Diff %d/%d (%f %%). First 10 mismatched rows:",
                         diff_rows,
                         total_rows,
                         diff_rows / total_rows * 100.0)
            logger.debug(mismatch_df[:20])
        else:
            logger.info("Results match validation dataset")

    # Now build the output
    return {
        "total_rows": total_rows,
        "matching_rows": int(comparison.count_matching_rows()),
        "diff_rows": diff_rows,
        "matching_cols": list(same_columns),
        "extra_cols": list(extra_columns),
        "missing_cols": list(missing_columns),
    }


def parse_args():
    argparser = argparse.ArgumentParser("Compares two data files which are parsable as Pandas dataframes")
    argparser.add_argument("data_files", nargs=2, help="Files to compare")
    argparser.add_argument('--include',
                           nargs='*',
                           help=("Which columns to include in the validation. "
                                 "Resulting columns is the intersection of all regex. Include applied before exclude"))
    argparser.add_argument(
        '--exclude',
        nargs='*',
        default=[r'^ID$', r'^_ts_'],
        help=("Which columns to exclude from the validation. "
              "Resulting ignored columns is the intersection of all regex. Include applied before exclude"))
    argparser.add_argument(
        '--index_col',
        help=("Specifies a column which will be used to align messages with rows in the validation dataset."))
    argparser.add_argument('--abs_tol',
                           type=float,
                           default=0.001,
                           help="Absolute tolerance to use when comparing float columns.")
    argparser.add_argument('--rel_tol',
                           type=float,
                           default=0.05,
                           help="Relative tolerance to use when comparing float columns.")
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    configure_logging(log_level=logging.DEBUG)

    df_a = read_file_to_df(args.data_files[0], file_type=FileTypes.Auto, df_type='pandas')
    df_b = read_file_to_df(args.data_files[1], file_type=FileTypes.Auto, df_type='pandas')
    results = compare_df(df_a,
                         df_b,
                         include_columns=args.include,
                         exclude_columns=args.exclude,
                         replace_idx=args.index_col,
                         abs_tol=args.abs_tol,
                         rel_tol=args.rel_tol)

    if results['diff_rows'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
