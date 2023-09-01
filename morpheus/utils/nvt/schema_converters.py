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

import dataclasses
import os
import typing
from functools import partial

import networkx as nx
import nvtabular as nvt
import pandas as pd
from merlin.core.dispatch import DataFrameType
from merlin.core.dispatch import annotate
from merlin.core.dispatch import is_dataframe_object
from merlin.core.dispatch import is_series_object
from merlin.dag import ColumnSelector
from nvtabular.ops import Filter
from nvtabular.ops import LambdaOp
from nvtabular.ops import Rename

import cudf

from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import DistinctIncrementColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn
from morpheus.utils.column_info import create_increment_col
from morpheus.utils.nvt.decorators import sync_df_as_pandas
from morpheus.utils.nvt.mutate import MutateOp
from morpheus.utils.nvt.transforms import json_flatten


@dataclasses.dataclass
class JSONFlattenInfo(ColumnInfo):
    """
    Subclass of `ColumnInfo`. Makes it easier to generate a graph of the column dependencies.

    Attributes
    ----------
    input_col_names : list
        List of input column names.
    output_col_names : list
        List of output column names.
    """

    input_col_names: list
    output_col_names: list

    def get_input_column_types(self) -> dict[str, str]:
        """
        Return a dictionary of input column names and types needed for processing. This is used for schema
        validation and should be overridden by subclasses.
        """
        return {name: ColumnInfo.convert_pandas_dtype(str) for name in self.input_col_names}


# Same in every way to the base, except we don't drop the index
class _MorpheusFilter(Filter):

    @annotate("Filter_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        filtered = self.f(df)
        if is_dataframe_object(filtered):
            new_df = filtered
        elif is_series_object(filtered) and filtered.dtype == bool:
            new_df = df[filtered]
        else:
            raise ValueError(f"Invalid output from filter op: f{filtered.__class__}")

        # new_df.reset_index(drop=True, inplace=True)
        return new_df


def _get_ci_column_selector(col_info) -> typing.Union[str, typing.List[str]]:
    """
    Return a column selector based on a ColumnInfo object.

    Parameters
    ----------
    col_info : ColumnInfo
        The ColumnInfo object.

    Returns
    -------
    Union[str, list of str]
        A column selector.

    Raises
    ------
    TypeError
        If the input `ci` is not an instance of ColumnInfo.
    Exception
        If the type of ColumnInfo is unknown.
    """

    if (not isinstance(col_info, ColumnInfo)):
        raise TypeError

    selected_cols = col_info.get_input_column_types()

    if (len(selected_cols) == 0 and isinstance(col_info, CustomColumn)):
        return ["*"]

    return list(selected_cols.keys())


def _json_flatten_from_input_schema(json_input_cols: typing.List[str],
                                    json_output_cols: typing.List[typing.Tuple[str, str]]) -> MutateOp:
    """
    Return a JSON flatten operation from an input schema.

    Parameters
    ----------
    json_input_cols : list of str
        A list of JSON input columns.
    json_output_cols : list of tuple
        A list of JSON output columns.

    Returns
    -------
    MutateOp
        A MutateOp object that represents the JSON flatten operation.
    """

    json_flatten_op = MutateOp(json_flatten, dependencies=json_input_cols, output_columns=json_output_cols)

    return json_flatten_op


@sync_df_as_pandas()
def _string_cat_col(df: pd.DataFrame, output_column: str, sep: str) -> pd.DataFrame:
    """
    Concatenate the string representation of all supplied columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    output_column : str
        The name of the output column.
    sep : str
        The separator to use when concatenating the strings.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame.
    """

    cat_col = df.apply(lambda row: sep.join(row.values.astype(str)), axis=1)

    return pd.DataFrame({output_column: cat_col}, index=cat_col.index)


# pylint
def _nvt_string_cat_col(
        column_selector: ColumnSelector,  # pylint: disable=unused-argument
        df: typing.Union[pd.DataFrame, cudf.DataFrame],
        output_column: str,
        input_columns: typing.List[str],
        sep: str = ', '):
    """
    Concatenates the string representation of the specified columns in a DataFrame.

    Parameters
    ----------
    column_selector : ColumnSelector
        A ColumnSelector object.
    df : Union[pandas.DataFrame, cudf.DataFrame]
        The input DataFrame.
    output_column : str
        The name of the output column.
    input_columns : list of str
        The input columns to concatenate.
    sep : str, default is ', '
        The separator to use when concatenating the strings.

    Returns
    -------
    Union[pandas.DataFrame, cudf.DataFrame]
        The resulting DataFrame.
    """

    return _string_cat_col(df[input_columns], output_column=output_column, sep=sep)


@sync_df_as_pandas()
def _increment_column(df: pd.DataFrame,
                      output_column: str,
                      input_column: str,
                      groupby_column: str,
                      period: str = 'D') -> pd.DataFrame:
    """
    Crete an increment a column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    output_column : str
        The name of the output column.
    input_column : str
        The name of the input column.
    period : str, default is 'D'
        The period to increment by.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame.
    """

    period_index = pd.to_datetime(df[input_column]).dt.to_period(period)
    groupby_col = df.groupby([groupby_column, period_index]).cumcount()

    return pd.DataFrame({output_column: groupby_col}, index=groupby_col.index)


def _nvt_increment_column(
        column_selector: ColumnSelector,  # pylint: disable=unused-argument
        df: typing.Union[pd.DataFrame, cudf.DataFrame],
        output_column: str,
        input_column: str,
        groupby_column: str,
        period: str = 'D') -> typing.Union[pd.DataFrame, cudf.DataFrame]:
    """
    Increment a column in a DataFrame.

    Parameters
    ----------
    column_selector : ColumnSelector
        A ColumnSelector object. Unused.
    df : Union[pandas.DataFrame, cudf.DataFrame]
        The input DataFrame.
    output_column : str
        The name of the output column.
    input_column : str
        The name of the input column.
    groupby_column : str
        Name of the column to groupby after creating the increment
    period : str, default is 'D'
        The period to increment by.

    Returns
    -------
    Union[pandas.DataFrame, cudf.DataFrame]
        The resulting DataFrame.
    """

    return _increment_column(df, output_column, input_column, groupby_column, period)


@sync_df_as_pandas()
def _distinct_increment_column(df: pd.DataFrame,
                               output_column: str,
                               input_column: str,
                               groupby_column: str = "username",
                               period: str = 'D',
                               timestamp_column: str = "timestamp") -> pd.DataFrame:

    output_series = create_increment_col(df=df,
                                         column_name=input_column,
                                         groupby_column=groupby_column,
                                         period=period,
                                         timestamp_column=timestamp_column)

    return pd.DataFrame({output_column: output_series}, index=output_series.index)


def _nvt_distinct_increment_column(_: ColumnSelector,
                                   df: typing.Union[pd.DataFrame, cudf.DataFrame],
                                   output_column: str,
                                   input_column: str,
                                   groupby_column: str = "username",
                                   period: str = 'D',
                                   timestamp_column: str = "timestamp") -> typing.Union[pd.DataFrame, cudf.DataFrame]:

    return _distinct_increment_column(df, output_column, input_column, groupby_column, period, timestamp_column)


@sync_df_as_pandas()
def _nvt_try_rename(df: pd.DataFrame, input_col_name: str, output_col_name: str, dtype: None) -> pd.Series:
    if (input_col_name in df.columns):
        return df.rename(columns={input_col_name: output_col_name})

    return pd.Series(None, index=df.index, dtype=dtype)


# Mappings from ColumnInfo types to functions that create the corresponding NVT operator
ColumnInfoProcessingMap = {
    BoolColumn:
        lambda ci,
        deps: [
            LambdaOp(
                lambda series: series.map(ci.value_map).astype(bool), dtype="bool", label=f"[BoolColumn] '{ci.name}'")
        ],
    ColumnInfo:
        lambda ci,
        deps: [
            MutateOp(lambda _,
                     df: df.assign(**{ci.name: df[ci.name].astype(ci.get_pandas_dtype())}) if (ci.name in df.columns)
                     else df.assign(**{ci.name: pd.Series(None, index=df.index, dtype=ci.get_pandas_dtype())}),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[ColumnInfo] '{ci.name}'")
        ],
    # Note(Devin): Custom columns are, potentially, very inefficient, because we have to run the custom function on the
    #   entire dataset this is because NVT requires the input column be available, but CustomColumn is a generic
    #   transform taking df->series(ci.name)
    CustomColumn:
        lambda ci,
        deps: [
            MutateOp(lambda _,
                     df: cudf.DataFrame({ci.name: ci.process_column_fn(df)}, index=df.index),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[CustomColumn] '{ci.name}'")
        ],
    DateTimeColumn:
        lambda ci,
        deps: [
            Rename(f=lambda name: ci.name if name == ci.input_name else name),
            LambdaOp(lambda series: series.astype(ci.dtype), dtype=ci.dtype, label=f"[DateTimeColumn] '{ci.name}'")
        ],
    IncrementColumn:
        lambda ci,
        deps: [
            MutateOp(partial(_nvt_increment_column,
                             output_column=ci.name,
                             input_column=ci.input_name,
                             groupby_column=ci.groupby_column,
                             period=ci.period),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[IncrementColumn] '{ci.input_name}.{ci.groupby_column}' => '{ci.name}'")
        ],
    DistinctIncrementColumn:
        lambda ci,
        deps: [
            MutateOp(partial(_nvt_distinct_increment_column,
                             output_column=ci.name,
                             input_column=ci.input_name,
                             groupby_column=ci.groupby_column,
                             period=ci.period,
                             timestamp_column=ci.timestamp_column),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=(f"[DistinctIncrementColumn] "
                            f"'{ci.input_name}.{ci.groupby_column}.{ci.timestamp_column}' => '{ci.name}'"))
        ],
    RenameColumn:
        lambda ci,
        deps: [
            MutateOp(lambda selector,
                     df: _nvt_try_rename(df, ci.input_name, ci.name, ci.dtype),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[RenameColumn] '{ci.input_name}' => '{ci.name}'")
        ],
    StringCatColumn:
        lambda ci,
        deps: [
            MutateOp(partial(_nvt_string_cat_col, output_column=ci.name, input_columns=ci.input_columns, sep=ci.sep),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[StringCatColumn] '{','.join(ci.input_columns)}' => '{ci.name}'")
        ],
    StringJoinColumn:
        lambda ci,
        deps: [
            MutateOp(partial(
                _nvt_string_cat_col, output_column=ci.name, input_columns=[ci.name, ci.input_name], sep=ci.sep),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[StringJoinColumn] '{ci.input_name}' => '{ci.name}'")
        ],
    JSONFlattenInfo:
        lambda ci,
        deps: [_json_flatten_from_input_schema(ci.input_col_names, ci.output_col_names)]
}


def _build_nx_dependency_graph(column_info_objects: typing.List[ColumnInfo]) -> nx.DiGraph:
    """
    Build a networkx directed graph for dependencies among columns.

    Parameters
    ----------
    column_info_objects : list of ColumnInfo
        List of column information objects.

    Returns
    -------
    nx.DiGraph
        A networkx DiGraph where nodes represent columns and edges represent dependencies between columns.

    """
    graph = nx.DiGraph()

    def _find_dependent_column(name, current_name):
        for col_info in column_info_objects:
            if col_info.name == current_name:
                continue

            # pylint: disable=no-else-return
            if col_info.name == name:
                return col_info
            elif col_info.__class__ == JSONFlattenInfo:
                if name in [c for c, _ in col_info.output_col_names]:
                    return col_info

        return None

    # For each column, determine the inputs for that column and add edges to the graph
    # Some columns will use simple strings
    for col_info in column_info_objects:
        graph.add_node(col_info.name)

        for input_col_name in col_info.get_input_column_types().keys():
            dep_col_info = _find_dependent_column(input_col_name, col_info.name)
            if (dep_col_info):
                graph.add_edge(dep_col_info.name, col_info.name)

    return graph


def _bfs_traversal_with_op_map(graph: nx.Graph,
                               ci_map: typing.Dict[str, ColumnInfo],
                               root_nodes: typing.List[typing.Any]):
    """
    Perform Breadth-First Search (BFS) on a given graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph on which BFS needs to be performed.
    ci_map : dict
        The dictionary mapping column info.
    root_nodes : list
        List of root nodes where BFS should start.

    Returns
    -------
    tuple
        Tuple containing the visited nodes and node-operation mapping.
    """

    visited = set()
    queue = list(root_nodes)
    node_op_map = {}

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)

            # We need to start an operator chain with a column selector, so root nodes need to prepend a parent
            # column selection operator
            parent_input = _get_ci_column_selector(ci_map[node])

            parents = list(graph.predecessors(node))

            # Thin the parent_input by any actual parents
            parent_input = [x for x in parent_input if x not in parents]

            # If we are a single element list, just use the element
            if (len(parent_input) == 1):
                parent_input = parent_input[0]

            if len(parents) > 0:
                # Not a root node, so we need to gather the parent operators, and collect them up.
                for parent in parents:
                    if isinstance(parent_input, list) and len(parent_input) == 0:
                        parent_input = node_op_map[parent]
                    else:
                        parent_input = parent_input + node_op_map[parent]

            # Map the column info object to its NVT operator implementation
            nvt_ops = ColumnInfoProcessingMap[type(ci_map[node])](ci_map[node], deps=[])

            # Chain ops together into a compound op
            node_op = parent_input
            for nvt_op in nvt_ops:
                node_op = node_op >> nvt_op

            # Set the op for this node to the compound operator
            node_op_map[node] = node_op

            # Add our neighbors to the queue
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                queue.append(neighbor)

    return visited, node_op_map


def _coalesce_leaf_nodes(node_op_map: typing.Dict[typing.Any, typing.Any],
                         column_info_objects: list[ColumnInfo]) -> typing.Any:
    """
    Coalesce (combine) operations for the leaf nodes of a graph.

    Parameters
    ----------
    node_op_map : dict
        Dictionary mapping nodes to operations.
    graph : nx.Graph
        The graph to be processed.
    preserve_re : regex
        Regular expression for nodes to be preserved.

    Returns
    -------
    obj
        Coalesced workflow for leaf nodes.
    """
    coalesced_workflow = None

    for column_info in column_info_objects:

        nvt_op = node_op_map[column_info.name]

        if coalesced_workflow is None:
            coalesced_workflow = nvt_op
        else:
            coalesced_workflow = coalesced_workflow + nvt_op

    return coalesced_workflow


def _coalesce_ops(graph: nx.Graph, column_info_objects: list[ColumnInfo]) -> typing.Any:
    """
    Coalesce (combine) operations for a graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be processed.
    ci_map : dict
        The dictionary mapping column info.
    preserve_re : regex, optional
        Regular expression for nodes to be preserved.

    Returns
    -------
    obj
        Coalesced workflow for the graph.
    """

    ci_map = {ci.name: ci for ci in column_info_objects}

    root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

    _, node_op_map = _bfs_traversal_with_op_map(graph, ci_map, root_nodes)

    coalesced_workflow = _coalesce_leaf_nodes(node_op_map, column_info_objects)

    return coalesced_workflow


def create_and_attach_nvt_workflow(input_schema: DataFrameInputSchema,
                                   visualize: typing.Optional[bool] = False) -> DataFrameInputSchema:
    """
    Converts an `input_schema` to a `nvt.Workflow` object.

    Parameters
    ----------
    input_schema : DataFrameInputSchema
        Input schema which specifies how the DataFrame should be processed.
    visualize : bool, optional
        If True, the resulting workflow graph will be visualized.
        Default is False.

    Returns
    -------
    nvt.Workflow
        A nvt.Workflow object representing the steps specified in the input schema.

    Raises
    ------
    ValueError
        If the input schema is empty.

    Notes
    -----
    First we aggregate all preprocessing steps, which we assume are independent of each other
    and can be run in parallel.

    Next we aggregate all column operations, which we assume are independent of each other and
    can be run in parallel and pass them the updated schema from the preprocessing steps.
    """

    if (input_schema is None):
        input_schema = DataFrameInputSchema()
        return input_schema
    if (len(input_schema.column_info) == 0):
        input_schema.nvt_workflow = None
        return input_schema

    # Note(Devin): soft locking problem with nvt operators, skip for now.
    #    column_info_objects.append(
    #        JSONFlattenInfo(input_col_names=list(json_cols),
    #                        output_col_names=json_output_cols,
    #                        dtype="str",
    #                        name="json_info"))

    graph = _build_nx_dependency_graph(input_schema.column_info)

    if os.getenv('MORPHEUS_NVT_VIS_DEBUG') is not None:
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos, with_labels=True, font_weight='bold')
        plt.show()

    coalesced_workflow = _coalesce_ops(graph, input_schema.column_info)
    if (input_schema.row_filter is not None):
        # Use our own filter here to preserve any index from the DataFrame
        coalesced_workflow = coalesced_workflow >> _MorpheusFilter(f=input_schema.row_filter)

    if (visualize):
        coalesced_workflow.graph.render(view=True, format='svg')

    input_schema.nvt_workflow = nvt.Workflow(coalesced_workflow)

    return input_schema
