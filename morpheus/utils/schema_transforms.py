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
from morpheus.utils.column_info import PreparedDFInfo
from morpheus.utils.nvt import patches
from morpheus.utils.nvt.extensions import morpheus_ext
from morpheus.utils.nvt.schema_converters import create_and_attach_nvt_workflow

if os.environ.get("MORPHEUS_IN_SPHINX_BUILD") is None:
    # Apply patches to NVT
    # TODO(Devin): Can be removed, once numpy mappings are updated in Merlin
    # ========================================================================
    patches.patch_numpy_dtype_registry()
    # ========================================================================

    # Add morpheus conversion mappings
    # ========================================================================
    morpheus_ext.register_morpheus_extensions()
    # =========================================================================

logger = logging.getLogger(__name__)


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

    If `input_schema` is an instance of `DataFrameInputSchema`, and it has a 'json_preproc' attribute,
    the function will first flatten the JSON columns and concatenate the results with the original DataFrame.

    Parameters
    ----------
    df_in : Union[pd.DataFrame, cudf.DataFrame]
        The input DataFrame to process.
    input_schema : Union[nvt.Workflow, DataFrameInputSchema]
        Defines the transformations to apply to 'df_in'.
        If an instance of nvt.Workflow, it is directly used to transform the dataframe.
        If an instance of DataFrameInputSchema, it is first converted to an nvt.Workflow,
        with JSON columns preprocessed if 'json_preproc' attribute is present.

    Returns
    -------
    Union[pd.DataFrame, cudf.DataFrame]
        The processed DataFrame. If 'df_in' was a pd.DataFrame, the return type is also pd.DataFrame,
        otherwise, it is cudf.DataFrame.

    Note
    ----
    Any transformation that needs to be performed should be defined in 'input_schema'.
    If 'df_in' is a pandas DataFrame, it is temporarily converted into a cudf DataFrame for the transformation.
    """

    convert_to_pd = False
    if (isinstance(df_in, pd.DataFrame)):
        convert_to_pd = True

    # If we're given an nvt_schema, we just use it.
    nvt_workflow = input_schema
    if (isinstance(input_schema, DataFrameInputSchema)):
        if (input_schema.nvt_workflow is None):
            input_schema = create_and_attach_nvt_workflow(input_schema)

        # Note(Devin): pre-flatten to avoid Dask hang when calling json_normalize within an NVT operator
        if (input_schema.prep_dataframe is not None):
            prepared_df_info: PreparedDFInfo = input_schema.prep_dataframe(df_in)

        nvt_workflow = input_schema.nvt_workflow

    preserve_df = None

    if prepared_df_info is not None:
        df_in = prepared_df_info.df

        if prepared_df_info.columns_to_preserve:
            preserve_df = df_in[prepared_df_info.columns_to_preserve]

    if (convert_to_pd):
        df_in = cudf.DataFrame(df_in)

    # NVT will always reset the index, so we need to save it and restore it after the transformation
    saved_index = df_in.index
    df_in.reset_index(drop=True, inplace=True)

    dataset = nvt.Dataset(df_in)

    if (nvt_workflow is not None):
        df_result = nvt_workflow.fit_transform(dataset).to_ddf().compute()
    else:
        df_result = df_in

    # Now reset the index
    if (len(df_result) == len(saved_index)):
        df_result.set_index(saved_index, inplace=True)
    else:
        # Must have done some filtering. Use the new index to index into the old index
        df_result.set_index(saved_index.take(df_result.index), inplace=True)

    if (convert_to_pd):
        df_result = df_result.to_pandas()

    # Restore preserved columns
    if (preserve_df is not None):
        # Ensure there is no overlap with columns to preserve
        columns_to_merge = set(preserve_df.columns) - set(df_result.columns)
        columns_to_merge = list(columns_to_merge)
        if (columns_to_merge):
            if (convert_to_pd):
                df_result = pd.concat([df_result, preserve_df[columns_to_merge]], axis=1)
            else:
                df_result = cudf.concat([df_result, preserve_df[columns_to_merge]], axis=1)

    return df_result
