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

import datetime
import io
import sys

import cupy as cp
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus import messages
from morpheus.messages import TensorMemory

# pylint: disable=unsupported-membership-test
# pylint: disable=unsubscriptable-object


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
    tasks["type_a"] = [{"key_x", "value_x"}]  # pylint: disable=unsupported-assignment-operation
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
    assert message.get_metadata("key_y") == "value_y"
    assert message.get_metadata("key_z") == "value_z"

    message.set_metadata("key_y", "value_yy")

    assert message.get_metadata()["key_y"] == "value_yy"

    message.get_metadata()["not_mutable"] = 5  # pylint: disable=unsupported-assignment-operation

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


def test_control_message_set_and_get_payload(dataset: DatasetManager):
    df = dataset["test_dataframe.jsonlines"]

    msg = messages.ControlMessage()
    payload = messages.MessageMeta(df)
    msg.payload(payload)

    payload2 = msg.payload()
    assert payload2 is not None

    DatasetManager.assert_df_equal(payload.df, payload2.df)


@pytest.mark.usefixtures("config_only_cpp")
def test_set_and_get_timestamp_single():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Define test data
    key = "group1::key1"
    timestamp = datetime.datetime.now()

    # Set timestamp
    msg.set_timestamp(key, timestamp)

    # Get timestamp and assert it's as expected
    result = msg.get_timestamp(key, True)
    assert result == timestamp, "The retrieved timestamp should match the one that was set."


@pytest.mark.usefixtures("config_only_cpp")
def test_filter_timestamp():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Setup test data
    group = "group1"
    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=1)
    msg.set_timestamp(f"{group}::key1", timestamp1)
    msg.set_timestamp(f"{group}::key2", timestamp2)

    # Use a regex that matches both keys
    result = msg.filter_timestamp(f"{group}::key.*")

    # Assert both keys are in the result and have correct timestamps
    assert len(result) == 2, "Both keys should be present in the result."
    assert result[f"{group}::key1"] == timestamp1, "The timestamp for key1 should match."
    assert result[f"{group}::key2"] == timestamp2, "The timestamp for key2 should match."


@pytest.mark.usefixtures("config_only_cpp")
def test_get_timestamp_fail_if_nonexist():
    # Create a ControlMessage instance
    msg = messages.ControlMessage()

    # Setup test data
    key = "nonexistent_key"

    # Attempt to get a timestamp for a non-existent key, expecting failure
    with pytest.raises(ValueError) as exc_info:
        msg.get_timestamp(key, True)
    assert str(exc_info.value) == "Timestamp for the specified key does not exist."


# Test setting and getting tensors with cupy arrays
@pytest.mark.usefixtures("config_only_cpp")
def test_tensors_setting_and_getting():
    data = {"input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1]), "segment_ids": cp.array([0, 0, 1])}
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=data["input_ids"].shape[0])
    tensor_memory.set_tensors(data)

    message.tensors(tensor_memory)

    retrieved_tensors = message.tensors()
    assert retrieved_tensors.count == data["input_ids"].shape[0], "Tensor count mismatch."

    for key, val in data.items():
        assert cp.allclose(retrieved_tensors.get_tensor(key), val), f"Mismatch in tensor data for {key}."


# Test retrieving tensor names and checking specific tensor existence
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_names_and_existence():
    tokenized_data = {
        "input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1]), "segment_ids": cp.array([0, 0, 1])
    }
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=tokenized_data["input_ids"].shape[0], tensors=tokenized_data)

    message.tensors(tensor_memory)
    retrieved_tensors = message.tensors()

    for key in tokenized_data:
        assert key in retrieved_tensors.tensor_names, f"Tensor {key} should be listed in tensor names."
        assert retrieved_tensors.has_tensor(key), f"Tensor {key} should exist."


# Test manipulating tensors after retrieval
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_manipulation_after_retrieval():
    tokenized_data = {
        "input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1]), "segment_ids": cp.array([0, 0, 1])
    }
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=3, tensors=tokenized_data)

    message.tensors(tensor_memory)

    retrieved_tensors = message.tensors()
    new_tensor = cp.array([4, 5, 6])
    retrieved_tensors.set_tensor("new_tensor", new_tensor)

    assert cp.allclose(retrieved_tensors.get_tensor("new_tensor"), new_tensor), "New tensor data mismatch."


# Assuming there's functionality to update all tensors at once
@pytest.mark.usefixtures("config_only_cpp")
def test_tensor_update():
    tokenized_data = {
        "input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1]), "segment_ids": cp.array([0, 0, 1])
    }
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=3, tensors=tokenized_data)

    message.tensors(tensor_memory)

    # Update tensors with new data
    new_tensors = {
        "input_ids": cp.array([4, 5, 6]), "input_mask": cp.array([1, 0, 1]), "segment_ids": cp.array([1, 1, 0])
    }

    tensor_memory.set_tensors(new_tensors)

    updated_tensors = message.tensors()

    for key, val in new_tensors.items():
        assert cp.allclose(updated_tensors.get_tensor(key), val), f"Mismatch in updated tensor data for {key}."


@pytest.mark.usefixtures("config_only_cpp")
def test_update_individual_tensor():
    initial_data = {"input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1])}
    update_data = {"input_ids": cp.array([4, 5, 6])}
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=3, tensors=initial_data)
    message.tensors(tensor_memory)

    # Update one tensor and retrieve all to ensure update integrity
    tensor_memory.set_tensor("input_ids", update_data["input_ids"])
    retrieved_tensors = message.tensors()

    # Check updated tensor
    assert cp.allclose(retrieved_tensors.get_tensor("input_ids"),
                       update_data["input_ids"]), "Input IDs update mismatch."
    # Ensure other tensor remains unchanged
    assert cp.allclose(retrieved_tensors.get_tensor("input_mask"),
                       initial_data["input_mask"]), "Input mask should remain unchanged after updating input_ids."


@pytest.mark.usefixtures("config_only_cpp")
def test_behavior_with_empty_tensors():
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=0)
    message.tensors(tensor_memory)

    retrieved_tensors = message.tensors()
    assert retrieved_tensors.count == 0, "Tensor count should be 0 for empty tensor memory."
    assert len(retrieved_tensors.tensor_names) == 0, "There should be no tensor names for empty tensor memory."


@pytest.mark.usefixtures("config_only_cpp")
def test_consistency_after_multiple_operations():
    initial_data = {"input_ids": cp.array([1, 2, 3]), "input_mask": cp.array([1, 1, 1])}
    message = messages.ControlMessage()
    tensor_memory = TensorMemory(count=3, tensors=initial_data)
    message.tensors(tensor_memory)

    # Update a tensor
    tensor_memory.set_tensor("input_ids", cp.array([4, 5, 6]))
    # Remove another tensor
    # Add a new tensor
    new_tensor = {"new_tensor": cp.array([7, 8, 9])}
    tensor_memory.set_tensor("new_tensor", new_tensor["new_tensor"])

    retrieved_tensors = message.tensors()
    assert retrieved_tensors.count == 3, "Tensor count mismatch after multiple operations."
    assert cp.allclose(retrieved_tensors.get_tensor("input_ids"),
                       cp.array([4, 5, 6])), "Mismatch in input_ids after update."
    assert cp.allclose(retrieved_tensors.get_tensor("new_tensor"),
                       new_tensor["new_tensor"]), "New tensor data mismatch."


class NonSerializablePyClass():

    def __init__(self):
        self.name = "non_serializable_py_class"

    def __getstate__(self):
        raise TypeError("This object is not serializable")


class NonSerializableNestedPyClass():

    def __init__(self):
        self.name = "non_serializable_nested_py_class"
        self.non_serializable = NonSerializablePyClass()


class NonSerializableNestedPyClassWithFile():

    def __init__(self):
        self.name = "non_serializable_nested_py_class_with_file"
        self.file_obj = io.StringIO("string data")


@pytest.fixture(name="py_object",
                scope="function",
                params=[NonSerializablePyClass, NonSerializableNestedPyClass, NonSerializableNestedPyClassWithFile])
def fixture_pyobject(request):
    return request.param()


@pytest.mark.usefixtures("config_only_cpp")
def test_metadata_holds_non_serializable_python_obj(py_object):

    message = messages.ControlMessage()

    obj = py_object
    key = obj.name

    message.set_metadata(key, obj)
    assert key in message.list_metadata()
    metadata = message.get_metadata(key)
    assert obj is metadata

    dict_with_obj = {"nested_obj": obj}
    message.set_metadata("nested", dict_with_obj)
    metadata_dict_with_obj = message.get_metadata("nested")

    # Check that the dict was serialized and recreated
    assert dict_with_obj is not metadata_dict_with_obj

    # Check that the nested non-serializable object is the same
    assert obj is metadata_dict_with_obj["nested_obj"]


@pytest.mark.usefixtures("config_only_cpp")
def test_tasks_hold_non_serializable_python_obj(py_object):

    message = messages.ControlMessage()

    obj = py_object
    task_key = "non_serializable"
    task_name = "task"

    message.add_task(task_key, {task_name: obj})
    assert message.has_task(task_key)
    task = message.get_tasks()[task_key][0][task_name]
    assert obj is task

    ref_count = sys.getrefcount(obj)
    assert message.remove_task(task_key)[task_name] is obj
    # Check the removed task decreases the reference count
    assert sys.getrefcount(obj) == ref_count - 1

    dict_with_obj = {"nested_obj": obj}
    message.set_metadata("nested", dict_with_obj)
    metadata_dict_with_obj = message.get_metadata("nested")

    # Check that the dict was serialized and recreated
    assert dict_with_obj is not metadata_dict_with_obj

    # Check that the nested non-serializable object is the same
    assert obj is metadata_dict_with_obj["nested_obj"]
