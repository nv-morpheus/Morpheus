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

    workflow = input_schema
    if (isinstance(input_schema, DataFrameInputSchema)):
        workflow = dataframe_input_schema_to_nvt_workflow(input_schema)

    convert_to_pd = False
    if (isinstance(df_in, pd.DataFrame)):
        convert_to_pd = True

        # for col in df_in.columns:
        #     print(df_in[col].dtype)
        #     if df_in[col].dtype == "datetime":
        #         df_in[col].dt.tz_localize(None)

        df_in = cudf.DataFrame(df_in)

    dataset = nvt.Dataset(df_in)

    result = workflow.fit_transform(dataset).to_ddf().compute()

    if (convert_to_pd):
        return result.to_pandas()

    return result
