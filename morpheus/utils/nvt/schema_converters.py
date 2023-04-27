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

import typing
from collections import deque

import cudf
import dataclasses
import json
import pandas as pd
import nvtabular as nvt
import networkx as nx
import matplotlib.pyplot as plt

from functools import partial
from merlin.dag import ColumnSelector
from nvtabular.ops import LambdaOp, Rename

from morpheus.utils.column_info import (BoolColumn, ColumnInfo, DataFrameInputSchema, DateTimeColumn,
                                        RenameColumn, StringCatColumn, StringJoinColumn, IncrementColumn)
from morpheus.utils.nvt import MutateOp, json_flatten
from nvtabular.workflow.node import WorkflowNode

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def sync_df_as_pandas(func: typing.Callable) -> typing.Callable:
    def wrapper(df: typing.Union[pd.DataFrame, cudf.DataFrame], **kwargs) -> typing.Union[
        pd.DataFrame, cudf.DataFrame]:
        convert_to_cudf = False
        if type(df) == cudf.DataFrame:
            convert_to_cudf = True
            df = df.to_pandas()

        df = func(df, **kwargs)

        if convert_to_cudf:
            df = cudf.from_pandas(df)

        return df

    return wrapper


def sync_series_as_pandas(func: typing.Callable) -> typing.Callable:
    def wrapper(series: typing.Union[pd.Series, cudf.Series], **kwargs) -> typing.Union[pd.Series, cudf.Series]:
        convert_to_cudf = False
        if (type(series) == cudf.Series):
            convert_to_cudf = True
            series = series.to_pandas()

        series = func(series, **kwargs)

        if (convert_to_cudf):
            series = cudf.from_pandas(series)

        return series

    return wrapper


@dataclasses.dataclass
class JSONFlattenInfo(ColumnInfo):
    """Subclass of `ColumnInfo`, Dummy ColumnInfo -- Makes it easier to generate a graph of the column dependencies"""
    input_col_names: list
    output_col_names: list


def resolve_json_output_columns(input_schema) -> typing.List[ColumnInfo]:
    column_info_objects = input_schema.column_info

    json_output_candidates = []
    for col_info in column_info_objects:
        json_output_candidates.append((col_info.name, col_info.dtype))
        if (hasattr(col_info, 'input_name')):
            json_output_candidates.append((col_info.input_name, col_info.dtype))
        if (hasattr(col_info, 'input_columns')):
            for col_name in col_info.input_columns:
                json_output_candidates.append((col_name, col_info.dtype))

    output_cols = []
    json_cols = input_schema.json_columns
    for col in json_output_candidates:
        cnsplit = col[0].split('.')
        if (len(cnsplit) > 1 and cnsplit[0] in json_cols):
            output_cols.append(col)

    return output_cols


def build_nx_dependency_graph(column_info_objects: typing.List[ColumnInfo]) -> nx.DiGraph:
    G = nx.DiGraph()

    def find_dependent_column(name, current_name):
        for ci in column_info_objects:
            if ci.name == current_name:
                continue
            if ci.name == name:
                return ci
            elif ci.__class__ == JSONFlattenInfo:
                if name in [c for c, _ in ci.output_col_names]:
                    return ci
        return None

    for col_info in column_info_objects:
        G.add_node(col_info.name)

        if col_info.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
            # If col_info.name != col_info.input_name then we're creating a potential dependency
            if col_info.name != col_info.input_name:
                dep_col_info = find_dependent_column(col_info.input_name, col_info.name)
                if dep_col_info:
                    # This CI is dependent on the dep_col_info CI
                    G.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == StringCatColumn:
            for input_col_name in col_info.input_columns:
                dep_col_info = find_dependent_column(input_col_name, col_info.name)
                if dep_col_info:
                    G.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == JSONFlattenInfo:
            for output_col_name in [c for c, _ in col_info.output_col_names]:
                dep_col_info = find_dependent_column(output_col_name, col_info.name)
                if dep_col_info:
                    G.add_edge(dep_col_info.name, col_info.name)

    return G


def get_ci_column_selector(ci: ColumnInfo):
    if ci.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
        return ci.input_name

    elif ci.__class__ == StringCatColumn:
        return ci.input_columns

    elif ci.__class__ == JSONFlattenInfo:
        return ci.input_col_names


def json_flatten_from_input_schema(json_input_cols, json_output_cols) -> MutateOp:
    json_flatten_op = MutateOp(json_flatten, dependencies=json_input_cols, output_columns=json_output_cols)

    return json_flatten_op


@sync_df_as_pandas
def string_cat_col(df: typing.Union[pd.DataFrame, cudf.DataFrame], output_column, sep) -> typing.Union[
    pd.DataFrame, cudf.DataFrame]:
    cat_col = df.apply(lambda row: sep.join(row.values.astype(str)), axis=1)

    return pd.DataFrame({output_column: cat_col})


def nvt_string_cat_col(column_selector: ColumnSelector, df: typing.Union[pd.DataFrame, cudf.DataFrame],
                       output_column, input_columns, sep: str = ', '):
    return string_cat_col(df[input_columns], output_column=output_column, sep=sep)


ColumnInfoProcessingMap = {
    BoolColumn: lambda ci, deps: [LambdaOp(lambda series: series.map(ci.value_map).astype(bool),
                                           dtype="bool")],
    ColumnInfo: lambda ci, deps: [LambdaOp(lambda series: series)],
    DateTimeColumn: lambda ci, deps: [Rename(f=lambda name: ci.name if name == ci.input_name else name),
                                      LambdaOp(lambda series: series.astype(ci.dtype),
                                               dtype=ci.dtype)],
    # RenameColumn: lambda ci, deps: [Rename(f=lambda name: ci.name if name == ci.input_name else name)],
    RenameColumn: lambda ci, deps: [MutateOp(lambda selector, df: df.rename(columns={ci.input_name: ci.name}),
                                             dependencies=deps, output_columns=[(ci.name, ci.dtype)])],
    StringCatColumn: lambda ci, deps: [MutateOp(
        partial(nvt_string_cat_col, output_column=ci.name, input_columns=ci.input_columns, sep=ci.sep),
        dependencies=deps, output_columns=[(ci.name, ci.dtype)])],
    JSONFlattenInfo: lambda ci, deps: [json_flatten_from_input_schema(ci.input_col_names, ci.output_col_names)]
}


def dfs_coalesce(graph, ci_map, parent, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    # Parent is guaranteed to have an input schema
    ops = ColumnInfoProcessingMap[type(ci_map[start])](ci_map[start], deps=[])
    for op in ops:
        parent = parent >> op

    ops = []
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            ops.extend(dfs_coalesce(graph, ci_map, parent, neighbor, visited))

    return ops or [parent]


def coalesce_ops(graph, ci_map):
    """Find nodes with no outgoing edges that are not dependent on the current node."""
    root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

    visited = set()
    coalesced_workflow = None
    for node in root_nodes:
        ci = ci_map[node]
        op_chain = get_ci_column_selector(ci)
        op_chain = dfs_coalesce(graph, ci_map, op_chain, node, visited)
        for op in op_chain:
            if (coalesced_workflow is None):
                coalesced_workflow = op
            else:
                coalesced_workflow = coalesced_workflow + op

    return coalesced_workflow


def input_schema_to_nvt_workflow(input_schema: DataFrameInputSchema) -> nvt.Workflow:
    """
    Converts an `input_schema` to a `nvt.Workflow` object

    First we aggregate all preprocessing steps, which we assume are independent of each other and can be run in parallel.

    Next we aggregate all column operations, which we assume are independent of each other and can be run in parallel and
    pass them the updated schema from the preprocessing steps.
    """

    # Try to guess which output columns we'll produce
    json_output_cols = resolve_json_output_columns(input_schema)

    json_cols = input_schema.json_columns
    column_info_objects = [ci for ci in input_schema.column_info]
    if (json_cols is not None and len(json_cols) > 0):
        column_info_objects.append(
            JSONFlattenInfo(input_col_names=[c for c in json_cols],
                            # output_col_names=[name for name, _ in json_output_cols],
                            output_col_names=json_output_cols,
                            dtype="str", name="json_info"))

        column_info_map = {ci.name: ci for ci in column_info_objects}

    graph = build_nx_dependency_graph(column_info_objects)

    # Uncomment to print the dependency layout
    # pos = graphviz_layout(graph, prog='neato')
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # plt.show()

    coalesced_workflow = coalesce_ops(graph, column_info_map)
    coalesced_workflow.graph.render(view=True, format='svg')

    return nvt.Workflow(coalesced_workflow)
