# Copyright (c) 2023, NVIDIA CORPORATION.
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

import os

import cudf

import morpheus


def _load_suffix_file():
    suffix_list_path = os.path.join(morpheus.DATA_DIR, "public_suffix_list.dat")
    # Read suffix list csv file
    suffix_df = cudf.io.csv.read_csv(suffix_list_path, names=["suffix"], header=None, dtype=["str"])
    suffix_df = suffix_df[suffix_df["suffix"].str.contains("^[^//]+$")]
    return suffix_df


_SUFFIX_DF = _load_suffix_file()
_ALLOWED_OUTPUT_COLS = {
    "hostname",
    "subdomain",
    "domain",
    "suffix",
}


def _handle_unknown_suffix(unknown_suffix_df, col_dict):
    if col_dict["hostname"]:
        unknown_suffix_df = unknown_suffix_df[["idx", "tld0"]]
        unknown_suffix_df = unknown_suffix_df.rename(columns={"tld0": "hostname"})
    else:
        unknown_suffix_df = unknown_suffix_df[["idx"]]

    if col_dict["subdomain"]:
        unknown_suffix_df["subdomain"] = ""
    if col_dict["domain"]:
        unknown_suffix_df["domain"] = ""
    if col_dict["suffix"]:
        unknown_suffix_df["suffix"] = ""

    return unknown_suffix_df


def _extract_tld(input_df, suffix_df, col_len, col_dict):
    tmp_dfs = []
    # Left join on single column dataframe does not provide expected results hence adding dummy column.
    suffix_df["dummy"] = ""
    # Iterating over each tld column starting from tld0 until it finds a match.
    for i in range(col_len + 1):
        # Add index to sort the parsed information with respect to input records order.
        cols_keep = ["idx"]
        tld_col = "tld" + str(i)
        suffix_df = suffix_df.rename(columns={suffix_df.columns[0]: tld_col})
        # Left join input_df with suffix_df on tld column for each iteration.
        merged_df = input_df.merge(suffix_df, on=tld_col, how="left")
        if i > 0:
            col_pos = i - 1
            # Retrieve records which satisfies join clause.
            joined_recs_df = merged_df[~merged_df["dummy"].isna()]
            if not joined_recs_df.empty:
                if col_dict["hostname"]:
                    joined_recs_df = joined_recs_df.rename(columns={"tld0": "hostname"})
                    cols_keep.append("hostname")
                if col_dict["subdomain"]:
                    cols_keep.append("subdomain")
                    joined_recs_df["subdomain"] = ""
                    if col_pos > 0:
                        for idx in range(0, col_pos):
                            joined_recs_df["subdomain"] = joined_recs_df["subdomain"].str.cat(joined_recs_df[idx],
                                                                                              sep=".")
                        joined_recs_df["subdomain"] = (joined_recs_df["subdomain"].str.replace(".^",
                                                                                               "").str.lstrip("."))
                if col_dict["domain"]:
                    joined_recs_df = joined_recs_df.rename(columns={col_pos: "domain"})
                    cols_keep.append("domain")
                if col_dict["suffix"]:
                    joined_recs_df = joined_recs_df.rename(columns={tld_col: "suffix"})
                    cols_keep.append("suffix")
                joined_recs_df = joined_recs_df[cols_keep]
                # Concat current iteration result to previous iteration result.
                tmp_dfs.append(joined_recs_df)
                # delete not required variable.
                del joined_recs_df
                # Assigning unprocessed records to input_df for next stage of processing.
                if i < col_len:
                    input_df = merged_df[merged_df["dummy"].isna()]
                    # Drop unwanted columns.
                    input_df = input_df.drop(["dummy", tld_col], axis=1)
                # Handles scenario when some records with last tld column matches to suffix list but not all.
                else:
                    merged_df = merged_df[merged_df["dummy"].isna()]
                    unknown_suffix_df = _handle_unknown_suffix(merged_df, col_dict)
                    tmp_dfs.append(unknown_suffix_df)
            # Handles scenario when all records with last tld column doesn't match to suffix list.
            elif i == col_len and not merged_df.empty:
                unknown_suffix_df = _handle_unknown_suffix(merged_df, col_dict)
                tmp_dfs.append(unknown_suffix_df)
            else:
                continue
    # Concat all temporary output dataframes
    output_df = cudf.concat(tmp_dfs)
    return output_df


def _create_col_dict(allowed_output_cols, req_cols):
    """Creates dictionary to apply check condition while extracting tld.
    """
    col_dict = {col: True for col in allowed_output_cols}
    if req_cols != allowed_output_cols:
        for col in allowed_output_cols ^ req_cols:
            col_dict[col] = False
    return col_dict


def _verify_req_cols(req_cols, allowed_output_cols):
    """Verify user requested columns against allowed output columns.
    """
    if req_cols is not None:
        if not req_cols.issubset(allowed_output_cols):
            raise ValueError(f"Given req_cols must be subset of {allowed_output_cols}")
    else:
        req_cols = allowed_output_cols
    return req_cols


def _generate_tld_cols(hostname_split_df, hostnames, col_len):
    hostname_split_df = hostname_split_df.fillna("")
    hostname_split_df["tld" + str(col_len)] = hostname_split_df[col_len]
    # Add all other elements of hostname_split_df
    for j in range(col_len - 1, 0, -1):
        hostname_split_df["tld" + str(j)] = (hostname_split_df[j].str.cat(hostname_split_df["tld" + str(j + 1)],
                                                                          sep=".").str.rstrip("."))
    # Assign hostname to tld0, to handle received input is just domain name.
    hostname_split_df["tld0"] = hostnames
    return hostname_split_df


def _extract_hostnames(urls):
    hostnames = urls.str.extract("([\\w]+[\\.].*[^/]|[\\-\\w]+[\\.].*[^/])")[0].str.extract("([\\w\\.\\-]+)")[0]
    return hostnames


def parse(urls, req_cols=None):
    """
    Extract hostname, domain, subdomain and suffix from URLs.

    Parameters
    ----------
    urls : cudf.Series
        URLs to be parsed.
    req_cols : typing.Set[str]
        Selected columns to extract. Can be subset of (hostname, domain, subdomain and suffix).

    Returns
    -------
    cudf.DataFrame
        Parsed dataframe with selected columns to extract.

    Examples
    --------
    >>> from cudf import DataFrame
    >>> from morpheus.parsers import url_parser
    >>>
    >>> input_df = DataFrame(
    ...     {
    ...         "url": [
    ...             "http://www.google.com",
    ...             "gmail.com",
    ...             "github.com",
    ...             "https://pandas.pydata.org",
    ...         ]
    ...     }
    ... )
    >>> url_parser.parse(input_df["url"])
                hostname  domain suffix subdomain
    0     www.google.com  google    com       www
    1          gmail.com   gmail    com
    2         github.com  github    com
    3  pandas.pydata.org  pydata    org    pandas
    >>> url_parser.parse(input_df["url"], req_cols={'domain', 'suffix'})
       domain suffix
    0  google    com
    1   gmail    com
    2  github    com
    3  pydata    org
    """
    req_cols = _verify_req_cols(req_cols, _ALLOWED_OUTPUT_COLS)
    col_dict = _create_col_dict(req_cols, _ALLOWED_OUTPUT_COLS)
    hostnames = _extract_hostnames(urls)
    url_index = urls.index
    del urls
    hostname_split_ser = hostnames.str.findall("([^.]+)")
    hostname_split_df = hostname_split_ser.to_frame()
    hostname_split_df = cudf.DataFrame(hostname_split_df[0].to_arrow().to_pylist())
    col_len = len(hostname_split_df.columns) - 1
    hostname_split_df = _generate_tld_cols(hostname_split_df, hostnames, col_len)
    # remove hostnames since they are available in hostname_split_df
    del hostnames
    # Assign input index to idx column.
    hostname_split_df["idx"] = url_index
    output_df = _extract_tld(hostname_split_df, _SUFFIX_DF, col_len, col_dict)
    # Sort index based on given input index order.
    output_df = output_df.sort_values("idx", ascending=True)
    # Drop temp columns.
    output_df = output_df.drop("idx", axis=1)
    # Reset the index.
    output_df = output_df.reset_index(drop=True)
    return output_df
