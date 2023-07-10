# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import typing

import nvtabular as nvt
import pandas as pd

import cudf

from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.nvt import dataframe_input_schema_to_nvt_workflow
# Apply patches to NVT
# TODO(Devin): Can be removed, once numpy mappings are updated in Merlin
# ========================================================================
from morpheus.utils.nvt.patches import patch_numpy_dtype_registry

patch_numpy_dtype_registry()
# ========================================================================

logger = logging.getLogger(__name__)


def _process_columns(df_in, input_schema: DataFrameInputSchema):
    # TODO(MDD): See what causes this to have such a perf impact over using df_in
    output_df = pd.DataFrame()

    convert_to_cudf = False

    if (isinstance(df_in, cudf.DataFrame)):
        df_in = df_in.to_pandas()
        convert_to_cudf = True

    # Iterate over the column info
    for ci in input_schema.column_info:
        try:
            output_df[ci.name] = ci._process_column(df_in)
        except Exception:
            logger.exception("Failed to process column '%s'. Dataframe: \n%s", ci.name, df_in, exc_info=True)
            raise

    if (input_schema.preserve_columns is not None):
        # Get the list of remaining columns not already added
        df_in_columns = set(df_in.columns) - set(output_df.columns)

        # Finally, keep any columns that match the preserve filters
        match_columns = [y for y in df_in_columns if input_schema.preserve_columns.match(y)]

        output_df[match_columns] = df_in[match_columns]

    if (convert_to_cudf):
        return cudf.from_pandas(output_df)

    return output_df


def _normalize_dataframe(df_in: pd.DataFrame, input_schema: DataFrameInputSchema):
    if (input_schema.json_columns is None or len(input_schema.json_columns) == 0):
        return df_in

    convert_to_cudf = False

    # Check if we are cudf
    if (isinstance(df_in, cudf.DataFrame)):
        df_in = df_in.to_pandas()
        convert_to_cudf = True

    json_normalized = []
    remaining_columns = list(df_in.columns)

    for j_column in input_schema.json_columns:

        if (j_column not in remaining_columns):
            continue

        normalized = pd.json_normalize(df_in[j_column])

        # Prefix the columns
        normalized.rename(columns={n: f"{j_column}.{n}" for n in normalized.columns}, inplace=True)

        # Reset the index otherwise there is a conflict
        normalized.reset_index(drop=True, inplace=True)

        json_normalized.append(normalized)

        # Remove from the list of remaining columns
        remaining_columns.remove(j_column)

    # Also need to reset the original index
    df_in.reset_index(drop=True, inplace=True)

    df_normalized = pd.concat([df_in[remaining_columns]] + json_normalized, axis=1)

    if (convert_to_cudf):
        return cudf.from_pandas(df_normalized)

    return df_normalized


def _filter_rows(df_in: pd.DataFrame, input_schema: DataFrameInputSchema):
    if (input_schema.row_filter is None):
        return df_in

    return input_schema.row_filter(df_in)


def process_dataframe(
    df_in: typing.Union[pd.DataFrame, cudf.DataFrame],
    input_schema: typing.Union[nvt.Workflow, DataFrameInputSchema],
) -> pd.DataFrame:
    """
    Applies column transformations as defined by `input_schema`
    """

    workflow = input_schema
    if (isinstance(input_schema, DataFrameInputSchema)):
        workflow = dataframe_input_schema_to_nvt_workflow(input_schema)

    convert_to_pd = False
    if (isinstance(df_in, pd.DataFrame)):
        convert_to_pd = True
        df_in = cudf.DataFrame(df_in)

    dataset = nvt.Dataset(df_in)

    result = workflow.fit_transform(dataset).to_ddf().compute()

    if (convert_to_pd):
        return result.to_pandas()
    else:
        return result
