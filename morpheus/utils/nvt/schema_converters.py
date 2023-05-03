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

from functools import partial
import typing

import cudf
import dataclasses
import pandas as pd

import morpheus.utils.nvt.cudf_dtype_mappings  # noqa: F401

from merlin.dag import ColumnSelector

from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn

from morpheus.utils.nvt import MutateOp
from morpheus.utils.nvt import json_flatten

import networkx as nx
import nvtabular as nvt
from nvtabular.ops import Filter
from nvtabular.ops import LambdaOp
from nvtabular.ops import Rename


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


@dataclasses.dataclass
class JSONFlattenInfo(ColumnInfo):
    """Subclass of `ColumnInfo`, Dummy ColumnInfo -- Makes it easier to generate a graph of the column dependencies"""
    input_col_names: list
    output_col_names: list


def resolve_json_output_columns(input_schema) -> typing.List[typing.Tuple[str, str]]:
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
    graph = nx.DiGraph()

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
        graph.add_node(col_info.name)

        if col_info.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
            # If col_info.name != col_info.input_name then we're creating a potential dependency
            if col_info.name != col_info.input_name:
                dep_col_info = find_dependent_column(col_info.input_name, col_info.name)
                if dep_col_info:
                    # This CI is dependent on the dep_col_info CI
                    graph.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == StringCatColumn:
            for input_col_name in col_info.input_columns:
                dep_col_info = find_dependent_column(input_col_name, col_info.name)
                if dep_col_info:
                    graph.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == JSONFlattenInfo:
            for output_col_name in [c for c, _ in col_info.output_col_names]:
                dep_col_info = find_dependent_column(output_col_name, col_info.name)
                if dep_col_info:
                    graph.add_edge(dep_col_info.name, col_info.name)

    return graph


def get_ci_column_selector(ci):
    if (ci.__class__ == ColumnInfo):
        return ci.name

    elif ci.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
        return ci.input_name

    elif ci.__class__ == StringCatColumn:
        return ci.input_columns

    elif ci.__class__ == JSONFlattenInfo:
        return ci.input_col_names

    else:
        raise Exception(f"Unknown ColumnInfo type: {ci.__class__}")


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


@sync_df_as_pandas
def increment_column(df: typing.Union[pd.DataFrame, cudf.DataFrame], output_column, input_column, period: str = 'D') -> \
        typing.Union[
            pd.DataFrame, cudf.DataFrame]:
    period_index = pd.to_datetime(df[input_column]).dt.to_period(period)
    groupby_col = df.groupby([output_column, period_index]).cumcount()

    return pd.DataFrame({output_column: groupby_col})


def nvt_increment_column(column_selector: ColumnSelector, df: typing.Union[pd.DataFrame, cudf.DataFrame],
                         output_column, input_column, period: str = 'D'):
    return increment_column(column_selector, df, output_column, input_column, period)


ColumnInfoProcessingMap = {
    BoolColumn: lambda ci, deps: [LambdaOp(lambda series: series.map(ci.value_map).astype(bool),
                                           dtype="bool", label=f"[BoolColumn] '{ci.name}'")],
    ColumnInfo: lambda ci, deps: [
        LambdaOp(lambda series: series.astype(ci.dtype), dtype=ci.dtype, label=f"[ColumnInfo] '{ci.name}'")],
    CustomColumn: lambda ci, deps: [
        MutateOp(lambda selector, df: ci.process_column_fn(df), dependencies=deps,
                 output_columns=[(ci.name, ci.dtype)]),
    ],
    DateTimeColumn: lambda ci, deps: [Rename(f=lambda name: ci.name if name == ci.input_name else name),
                                      LambdaOp(lambda series: series.astype(ci.dtype),
                                               dtype=ci.dtype, label=f"[DateTimeColumn] '{ci.name}'")],
    IncrementColumn: lambda ci, deps: [
        MutateOp(partial(nvt_increment_column, output_column=ci.groupby_column, input_column=ci.name, period=ci.period),
                 dependencies=deps, output_columns=[(ci.name, ci.groupby_column)],
                 label=f"[IncrementColumn] '{ci.name}' => '{ci.groupby_column}'")],
    RenameColumn: lambda ci, deps: [MutateOp(lambda selector, df: df.rename(columns={ci.input_name: ci.name}),
                                             dependencies=deps, output_columns=[(ci.name, ci.dtype)],
                                             label=f"[RenameColumn] '{ci.input_name}' => '{ci.name}'")],
    StringCatColumn: lambda ci, deps: [MutateOp(
        partial(nvt_string_cat_col, output_column=ci.name, input_columns=ci.input_columns, sep=ci.sep),
        dependencies=deps, output_columns=[(ci.name, ci.dtype)],
        label=f"[StringCatColumn] '{','.join(ci.input_columns)}' => '{ci.name}'")],
    StringJoinColumn: lambda ci, deps: [MutateOp(
        partial(nvt_string_cat_col, output_column=ci.name, input_columns=[ci.name, ci.input_name], sep=ci.sep),
        dependencies=deps, output_columns=[(ci.name, ci.dtype)],
        label=f"[StringJoinColumn] '{ci.input_name}' => '{ci.name}'")],
    JSONFlattenInfo: lambda ci, deps: [json_flatten_from_input_schema(ci.input_col_names, ci.output_col_names)]
}


def bfs_traversal_with_op_map(graph, ci_map, root_nodes):
    visited = set()
    queue = [n for n in root_nodes]
    node_op_map = {}

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)

            parents = [n for n in graph.predecessors(node)]
            if len(parents) == 0:
                parent_input = get_ci_column_selector(ci_map[node])
            else:
                parent_input = None
                for parent in parents:
                    if parent_input is None:
                        parent_input = node_op_map[parent]
                    else:
                        parent_input = parent_input + node_op_map[parent]

            ops = ColumnInfoProcessingMap[type(ci_map[node])](ci_map[node], deps=[])
            node_op = parent_input
            for op in ops:
                node_op = node_op >> op

            node_op_map[node] = node_op

            neighbors = [n for n in graph.neighbors(node)]
            for neighbor in neighbors:
                queue.append(neighbor)

    return visited, node_op_map


def coalesce_leaf_nodes(node_op_map, graph, preserve_re):
    coalesced_workflow = None
    for node, op in node_op_map.items():
        neighbors = [n for n in graph.neighbors(node)]
        # Only add the operators for leaf nodes, or those explicitly preserved
        if len(neighbors) == 0 or (preserve_re and preserve_re.match(node)):
            if coalesced_workflow is None:
                coalesced_workflow = op
            else:
                coalesced_workflow = coalesced_workflow + op

    return coalesced_workflow


def coalesce_ops(graph, ci_map, preserve_re=None):
    root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
    visited, node_op_map = bfs_traversal_with_op_map(graph, ci_map, root_nodes)
    coalesced_workflow = coalesce_leaf_nodes(node_op_map, graph, preserve_re=preserve_re)

    return coalesced_workflow


def dataframe_input_schema_to_nvt_workflow(input_schema: DataFrameInputSchema, visualize=False) -> nvt.Workflow:
    """
    Converts an `input_schema` to a `nvt.Workflow` object

    First we aggregate all preprocessing steps, which we assume are independent of each other and can be run in parallel.

    Next we aggregate all column operations, which we assume are independent of each other and can be run in parallel and
    pass them the updated schema from the preprocessing steps.
    """

    if (input_schema is None or len(input_schema.column_info) == 0):
        raise ValueError("Input schema is empty")

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
    # from matplotlib import pyplot as plt
    # pos = graphviz_layout(graph, prog='neato')
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # plt.show()

    coalesced_workflow = coalesce_ops(graph, column_info_map, preserve_re=input_schema.preserve_columns)
    if (input_schema.row_filter is not None):
        coalesced_workflow = coalesced_workflow >> Filter(f=input_schema.row_filter)

    # Uncomment to display the NVT workflow render
    if (visualize):
        coalesced_workflow.graph.render(view=True, format='svg')

    return nvt.Workflow(coalesced_workflow)
