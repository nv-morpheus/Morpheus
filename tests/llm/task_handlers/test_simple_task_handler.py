# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from _utils.dataset_manager import DatasetManager
from _utils.llm import execute_task_handler
# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import MessageMeta as MessageMetaCpp
# pylint: enable=morpheus-incorrect-lib-from-import
from morpheus.llm import LLMTaskHandler
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage


def test_constructor():
    task_handler = SimpleTaskHandler()
    assert isinstance(task_handler, LLMTaskHandler)


@pytest.mark.parametrize("output_columns,expected_input_names",
                         [(None, ["response"]), (['frogs', 'toads', 'lizards'], ['frogs', 'toads', 'lizards'])])
def test_get_input_names(output_columns: list[str] | None, expected_input_names: list[str]):
    task_handler = SimpleTaskHandler(output_columns=output_columns)
    assert task_handler.get_input_names() == expected_input_names


def test_try_handle(dataset_cudf: DatasetManager):
    reptiles = ['lizards', 'snakes', 'turtles', 'frogs', 'toads']
    df = dataset_cudf["filter_probs.csv"][0:5]  # Take the first 5 rows since there are only have 5 reptiles
    expected_df = df.copy(deep=True)
    expected_df['reptiles'] = reptiles.copy()

    message = ControlMessage()
    message.payload(MessageMetaCpp(df))

    task_handler = SimpleTaskHandler(['reptiles'])

    task_dict = {"input_keys": ["reptiles"]}
    response_messages = execute_task_handler(task_handler, task_dict, message, reptiles=reptiles)
    assert len(response_messages) == 1
    response_message = response_messages[0]

    dataset_cudf.assert_compare_df(df, response_message.payload().df)
