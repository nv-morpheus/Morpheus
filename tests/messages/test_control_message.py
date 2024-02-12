#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cudf
import cupy as cp
import pytest

from morpheus import messages
from morpheus._lib.messages import TensorMemory as CppTensorMemory


# Fixture to create example tokenized data using cupy arrays
@pytest.fixture(scope="function")
def tokenized_data():
    return {
        "input_ids": cp.array([1, 2, 3]),
        "input_mask": cp.array([1, 1, 1]),
        "segment_ids": cp.array([0, 0, 1])
    }


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_init():
    messages.ControlMessage()  # noqa: F841
    messages.ControlMessage({"test": "test"})  # noqa: F841


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_tasks():
    message = messages.ControlMessage()
    assert len(message.get_tasks()) == 0

    # Ensure a single task can be read
    message = messages.ControlMessage()
    message.add_task("type_a", {"key_x": "value_x"})
    assert len(message.get_tasks()) == 1
    assert "type_a" in message.get_tasks()
    assert len(message.get_tasks()["type_a"]) == 1
    assert message.get_tasks()["type_a"][0]["key_x"] == "value_x"

    # Ensure multiple task types of different types can be read
    message = messages.ControlMessage()
    message.add_task("type_a", {"key_x": "value_x"})
    message.add_task("type_b", {"key_y": "value_y"})
    assert len(message.get_tasks()) == 2
    assert "type_a" in message.get_tasks()
    assert len(message.get_tasks()["type_a"]) == 1
    assert message.get_tasks()["type_a"][0]["key_x"] == "value_x"
    assert "type_b" in message.get_tasks()
    assert len(message.get_tasks()["type_b"]) == 1
    assert message.get_tasks()["type_b"][0]["key_y"] == "value_y"

    # Ensure multiple task types of the same type can be read
    message = messages.ControlMessage()
    message.add_task("type_a", {"key_x": "value_x"})
    message.add_task("type_a", {"key_y": "value_y"})
    assert len(message.get_tasks()) == 1
    assert "type_a" in message.get_tasks()
    assert len(message.get_tasks()["type_a"]) == 2
    assert message.get_tasks()["type_a"][0]["key_x"] == "value_x"
    assert message.get_tasks()["type_a"][1]["key_y"] == "value_y"

    # Ensure the underlying tasks cannot are not modified
    message = messages.ControlMessage()
    tasks = message.get_tasks()
    tasks["type_a"] = [{"key_x", "value_x"}]
    assert len(message.get_tasks()) == 0

    message = messages.ControlMessage()
    message.add_task("type_a", {"key_x": "value_x"})
    message.add_task("type_a", {"key_y": "value_y"})
    assert len(message.get_tasks()) == 1
    assert "type_a" in message.get_tasks()
    assert len(message.get_tasks()["type_a"]) == 2
    assert message.get_tasks()["type_a"][0]["key_x"] == "value_x"
    assert message.get_tasks()["type_a"][1]["key_y"] == "value_y"


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_metadata():
    message = messages.ControlMessage()

    message.set_metadata("key_x", "value_x")
    message.set_metadata("key_y", "value_y")
    message.set_metadata("key_z", "value_z")

    metadata_tags = message.list_metadata()
    assert len(metadata_tags) == 3

    assert "key_x" in metadata_tags
    assert "key_y" in metadata_tags
    assert "key_z" in metadata_tags
    assert message.get_metadata("key_x") == "value_x"
    assert message.get_metadata()["key_y"] == "value_y"
    assert message.get_metadata("key_z") == "value_z"

    message.set_metadata("key_y", "value_yy")

    assert message.get_metadata()["key_y"] == "value_yy"

    message.get_metadata()["not_mutable"] = 5

    assert "not_mutable" not in message.get_metadata()


def test_set_and_get_metadata():
    message = messages.ControlMessage()

    # Test setting and getting metadata
    message.set_metadata("test_key", "test_value")
    assert message.get_metadata("test_key") == "test_value"

    # Test getting metadata with a default value when the key does not exist
    default_value = "default"
    assert message.get_metadata("nonexistent_key", default_value) == default_value

    # Test getting all metadata
    message.set_metadata("another_key", "another_value")
    all_metadata = message.get_metadata()
    assert isinstance(all_metadata, dict)
    assert all_metadata["test_key"] == "test_value"
    assert all_metadata["another_key"] == "another_value"


def test_list_metadata():
    message = messages.ControlMessage()

    # Setting some metadata
    message.set_metadata("key1", "value1")
    message.set_metadata("key2", "value2")
    message.set_metadata("key3", "value3")

    # Listing all metadata keys
    keys = message.list_metadata()
    assert isinstance(keys, list)
    assert set(keys) == {"key1", "key2", "key3"}


def test_get_metadata_default_value():
    message = messages.ControlMessage()

    # Setting metadata to test default value retrieval
    message.set_metadata("existing_key", "existing_value")

    # Getting an existing key without default value
    assert message.get_metadata("existing_key") == "existing_value"

    # Getting a non-existing key with default value provided
    assert message.get_metadata("non_existing_key", "default_value") == "default_value"


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_get():
    raw_control_message = messages.ControlMessage({
        "test": "test_rcm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })
    control_message = messages.ControlMessage({
        "test": "test_cm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })

    assert "test" not in raw_control_message.config()
    assert (raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert (control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set():
    raw_control_message = messages.ControlMessage()
    control_message = messages.ControlMessage()

    raw_control_message.config({
        "test": "test_rcm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })
    control_message.config({"test": "test_cm", "tasks": [{"type": "load", "properties": {"loader_id": "payload"}}]})

    assert "test" not in raw_control_message.config()
    assert (raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert (control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set_and_get_payload():
    df = cudf.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [True, False, True, False, True]
    })

    msg = messages.ControlMessage()
    payload = messages.MessageMeta(df)
    msg.payload(payload)

    payload2 = msg.payload()
    assert payload2 is not None
    assert payload.df == payload2.df


@pytest.mark.usefixtures("config_only_cpp")
def test_set_and_get_timestamp_single():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Define test data
    group = "group1"
    key = "key1"
    timestamp_ns = 123456789

    # Set timestamp
    msg.set_timestamp(group, key, timestamp_ns)

    # Get timestamp and assert it's as expected
    result = msg.get_timestamp(group, key, True)
    assert result == timestamp_ns, "The retrieved timestamp should match the one that was set."


@pytest.mark.usefixtures("config_only_cpp")
def test_get_timestamp_with_regex():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Setup test data
    group = "group1"
    msg.set_timestamp(group, "key1", 100)
    msg.set_timestamp(group, "key2", 200)

    # Use a regex that matches both keys
    result = msg.get_timestamp(group, "key.*")

    # Assert both keys are in the result and have correct timestamps
    assert len(result) == 2, "Both keys should be present in the result."
    assert result[f"{group}::key1"] == 100, "The timestamp for key1 should be 100."
    assert result[f"{group}::key2"] == 200, "The timestamp for key2 should be 200."


@pytest.mark.usefixtures("config_only_cpp")
def test_get_timestamp_fail_if_nonexist():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Setup test data
    group = "group2"
    key = "nonexistent_key"

    # Attempt to get a timestamp for a non-existent key, expecting failure
    try:
        msg.get_timestamp(group, key, True)
        assert False, "Expected a ValueError for a non-existent key when fail_if_nonexist is True."
    except ValueError as e:
        assert str(e) == "Timestamp for the specified key does not exist."


# Test setting and getting tensors with cupy arrays
@pytest.mark.usefixtures("config_only_cpp")
def test_tensors_setting_and_getting():
    data = {
        "input_ids": cp.array([1, 2, 3]),
        "input_mask": cp.array([1, 1, 1]),
        "segment_ids": cp.array([0, 0, 1])
    }
    message = messages.ControlMessage()
    tensor_memory = CppTensorMemory(count=data["input_ids"].shape[0],
                                    tensors=data)

    message.tensors(tensor_memory)

    retrieved_tensors = message.tensors()
    assert retrieved_tensors.count == data["input_ids"].shape[0], "Tensor count mismatch."

    for key in data:
        assert cp.allclose(retrieved_tensors.get_tensor(key),
                           data[key]), f"Mismatch in tensor data for {key}."


# Test retrieving tensor names and checking specific tensor existence
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_names_and_existence(tokenized_data):
    message = messages.ControlMessage()
    tensor_memory = CppTensorMemory(count=tokenized_data["input_ids"].shape[0],
                                    tensors=tokenized_data)

    message.tensors(tensor_memory)
    retrieved_tensors = message.tensors()

    for key in tokenized_data:
        assert key in retrieved_tensors.tensor_names, f"Tensor {key} should be listed in tensor names."
        assert retrieved_tensors.has_tensor(key), f"Tensor {key} should exist."


# Test manipulating tensors after retrieval
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_manipulation_after_retrieval(tokenized_data):
    message = messages.ControlMessage()
    tensor_memory = CppTensorMemory(count=3, tensors=tokenized_data)

    message.tensors(tensor_memory)

    retrieved_tensors = message.tensors()
    new_tensor = cp.array([4, 5, 6])
    retrieved_tensors.set_tensor("new_tensor", new_tensor)

    assert cp.allclose(retrieved_tensors.get_tensor("new_tensor"), new_tensor), "New tensor data mismatch."


# Assuming there's functionality to update all tensors at once
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_update(tokenized_data):
    message = messages.ControlMessage()
    tensor_memory = CppTensorMemory(count=3, tensors=tokenized_data)

    message.tensors(tensor_memory)

    # Update tensors with new data
    new_tensors = {
        "input_ids": cp.array([4, 5, 6]),
        "input_mask": cp.array([1, 0, 1]),
        "segment_ids": cp.array([1, 1, 0])
    }
    tensor_memory.set_tensors(new_tensors)
    updated_tensors = message.tensors()

    for key in new_tensors:
        assert cp.allclose(updated_tensors.get_tensor(key),
                           new_tensors[key]), f"Mismatch in updated tensor data for {key}."


if (__name__ == "__main__"):
    test_control_message_init()
    test_control_message_get()
    test_control_message_set()
    test_control_message_set_and_get_payload()
