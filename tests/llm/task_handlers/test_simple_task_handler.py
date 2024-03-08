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

import pytest

from _utils.dataset_manager import DatasetManager
from _utils.llm import execute_task_handler
from morpheus.llm import LLMTaskHandler
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.type_aliases import DataFrameType


def test_constructor():
    task_handler = SimpleTaskHandler()
    assert isinstance(task_handler, LLMTaskHandler)


@pytest.mark.parametrize("output_columns,expected_input_names",
                         [(None, ["response"]), (['frogs', 'toads', 'lizards'], ['frogs', 'toads', 'lizards'])])
def test_get_input_names(output_columns: list[str] | None, expected_input_names: list[str]):
    task_handler = SimpleTaskHandler(output_columns=output_columns)
    assert task_handler.get_input_names() == expected_input_names


@pytest.mark.parametrize("row_mask", [None, [True, False, True, False, True]])
def test_execute_task_handler(dataset_cudf: DatasetManager, row_mask: list[bool] | None):
    reptiles = ['lizards', 'snakes', 'turtles', 'frogs', 'toads']
    if row_mask is not None:
        expected_reptiles: list[str] = []
        for (i, r) in enumerate(reptiles):
            if row_mask[i]:
                expected_reptiles.append(r)
            else:
                expected_reptiles.append('')
    else:
        expected_reptiles = reptiles.copy()

    # Take the first 5 rows since there are only have 5 reptiles
    df: DataFrameType = dataset_cudf["filter_probs.csv"][0:5]

    expected_df = df.copy(deep=True)
    expected_df['reptiles'] = expected_reptiles

    df['reptiles'] = ''  # init a new column of empty strings

    message = ControlMessage()
    message.payload(MessageMeta(df))

    task_handler = SimpleTaskHandler(['reptiles'])

    task_dict = {"input_keys": ["reptiles"]}
    response_messages = execute_task_handler(task_handler, task_dict, message, reptiles=reptiles, row_mask=row_mask)
    assert len(response_messages) == 1
    response_message = response_messages[0]

    dataset_cudf.assert_compare_df(expected_df, response_message.payload().df)
