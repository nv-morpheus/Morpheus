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
import os
import typing

import nvtabular as nvt
import pandas as pd

import cudf

from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.nvt import register_morpheus_extensions
from morpheus.utils.nvt.patches import patch_numpy_dtype_registry
from morpheus.utils.nvt.schema_converters import dataframe_input_schema_to_nvt_workflow

if os.environ.get("MORPHEUS_IN_SPHINX_BUILD") is None:
    # Apply patches to NVT
    # TODO(Devin): Can be removed, once numpy mappings are updated in Merlin
    # ========================================================================
    patch_numpy_dtype_registry()
    # ========================================================================

    # Add morpheus conversion mappings
    # ========================================================================
    register_morpheus_extensions()
    # =========================================================================

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


@typing.overload
def process_dataframe(
        df_in: pd.DataFrame,
        input_schema: typing.Union[nvt.Workflow, DataFrameInputSchema],
) -> pd.DataFrame:
    ...


@typing.overload
def process_dataframe(
        df_in: cudf.DataFrame,
        input_schema: typing.Union[nvt.Workflow, DataFrameInputSchema],
) -> cudf.DataFrame:
    ...


def process_dataframe(
        df_in: typing.Union[pd.DataFrame, cudf.DataFrame],
        input_schema: typing.Union[nvt.Workflow, DataFrameInputSchema],
) -> typing.Union[pd.DataFrame, cudf.DataFrame]:
    """
    Applies column transformations to the input dataframe as defined by the `input_schema`.

    Parameters
    ----------
    df_in : Union[pd.DataFrame, cudf.DataFrame]
        The input DataFrame to process.
    input_schema : Union[nvt.Workflow, DataFrameInputSchema]
        If an instance of nvt.Workflow, it is directly used to transform the dataframe.
        If an instance of DataFrameInputSchema, it is converted to a nvt.Workflow before being used.

    Returns
    -------
    Union[pd.DataFrame, cudf.DataFrame]
        The processed DataFrame. If 'df_in' was a pd.DataFrame, the return type is pd.DataFrame.
        Otherwise, it is cudf.DataFrame.
    """

    work_algorithm = {
        "data_frame_input_schema": None,
        "nvt_workflow": None
    }
    workflow = input_schema
    if (isinstance(input_schema, DataFrameInputSchema)):
        work_algorithm["data_frame_input_schema"] = workflow
        workflow = dataframe_input_schema_to_nvt_workflow(input_schema)

    work_algorithm["nvt_workflow"] = workflow

    convert_to_pd = False
    if (isinstance(df_in, pd.DataFrame)):
        convert_to_pd = True

        df_in = cudf.DataFrame(df_in)

    if (df_in.shape[0] < 500 and work_algorithm["data_frame_input_schema"] is not None):
        input_schema = work_algorithm["data_frame_input_schema"]
        df_result = _normalize_dataframe(df_in, input_schema)
        df_result = _process_columns(df_result, input_schema)
        df_result = _filter_rows(df_result, input_schema)
    else:
        nvt_workflow = work_algorithm["nvt_workflow"]
        dataset = nvt.Dataset(df_in)

        df_result = nvt_workflow.fit_transform(dataset).to_ddf().compute()

    if (convert_to_pd):
        return df_result.to_pandas()

    return df_result
