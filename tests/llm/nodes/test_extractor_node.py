# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

import pytest

import cudf

from _utils.llm import execute_node
from morpheus.llm import LLMNodeBase
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.type_aliases import DataFrameType


def test_constructor():
    node = ExtracterNode()
    assert isinstance(node, LLMNodeBase)


def test_get_input_names():
    node = ExtracterNode()
    assert len(node.get_input_names()) == 0


@pytest.mark.parametrize("use_filter_fn", [False, True])
def test_execute(use_filter_fn: bool):
    insects = ["ant", "bee", "butterfly", "mosquito", "grasshopper"]
    mammals = ["lion", "dolphin", "gorilla", "wolf", "tiger"]
    reptiles = ['lizards', 'snakes', 'turtles', 'frogs', 'toads']
    df = cudf.DataFrame({"insects": insects.copy(), "mammals": mammals.copy(), "reptiles": reptiles.copy()})
    message = ControlMessage()
    message.payload(MessageMeta(df))

    if use_filter_fn:

        def filter_fn(df: DataFrameType) -> typing.Iterable[bool]:
            return df['insects'].str.startswith('b')

        expected_mammals = ["dolphin", "gorilla"]
        expected_repitles = ['snakes', 'turtles']
    else:
        filter_fn = None
        expected_mammals = mammals.copy()
        expected_repitles = reptiles.copy()

    expected_output = {"mammals": expected_mammals, "reptiles": expected_repitles}

    task_dict = {"input_keys": ["mammals", "reptiles"]}
    node = ExtracterNode(filter_fn=filter_fn)
    assert execute_node(node, task_dict=task_dict, input_message=message) == expected_output
