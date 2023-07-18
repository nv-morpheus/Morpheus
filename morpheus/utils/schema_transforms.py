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

import json
import logging
import os
import typing

import nvtabular as nvt
import pandas as pd

import cudf

from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.nvt import register_morpheus_extensions
from morpheus.utils.nvt.patches import patch_numpy_dtype_registry
from morpheus.utils.nvt.schema_converters import create_and_attach_nvt_workflow

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
            df_in = input_schema.prep_dataframe(df_in)

        nvt_workflow = input_schema.nvt_workflow

    if (convert_to_pd):
        df_in = cudf.DataFrame(df_in)

    dataset = nvt.Dataset(df_in)

    if (nvt_workflow is not None):
        df_result = nvt_workflow.fit_transform(dataset).to_ddf().compute()
    else:
        df_result = df_in

    if (convert_to_pd):
        return df_result.to_pandas()

    return df_result
