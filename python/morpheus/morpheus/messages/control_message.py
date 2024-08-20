# Copyright (c) 2024, NVIDIA CORPORATION.
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
import logging
import re
import typing
from collections import defaultdict
from collections import deque
from datetime import datetime

# Users of this module should import ControlMessageType from morpheus.messages, we can't do that here without causing a
# circular import error, instead we import it from the _lib module, we don't want to put `_messages.ControlMessageType`
# in the public API and confuse users
import morpheus._lib.messages as _messages
from morpheus._lib.messages import ControlMessageType  # pylint: disable=morpheus-incorrect-lib-from-import
from morpheus.cli.utils import get_enum_keys
from morpheus.cli.utils import get_enum_members
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_base import MessageBase
from morpheus.messages.message_meta import MessageMeta

logger = logging.getLogger(__name__)


@dataclasses.dataclass(init=False)
class ControlMessage(MessageBase, cpp_class=_messages.ControlMessage):

    def __init__(self, config: dict = None):
        super().__init__()

        self._config: dict = {"metadata": {}}

        self._payload: MessageMeta = None
        self._tensors: TensorMemory = None

        self._tasks: dict[str, deque] = defaultdict(deque)
        self._timestamps: dict[str, datetime] = {}
        self._type: ControlMessageType = ControlMessageType.NONE

        self.config(config)

    def config(self, config: dict = None) -> dict:
        if config is not None:
            cm_type: str | ControlMessageType = config.get("type")
            if cm_type is not None:
                if isinstance(cm_type, str):
                    try:
                        cm_type = get_enum_members(ControlMessageType)[cm_type]
                    except KeyError as exc:
                        raise ValueError(
                            f"Invalid ControlMessageType: {cm_type}, supported types: {get_enum_keys(ControlMessageType)}"
                        ) from exc

                self._type = cm_type

            tasks = config.get("tasks")
            if tasks is not None:
                for task in tasks:
                    self.add_task(task["type"], task["properties"])

            self._config = {"metadata": config.get("metadata", {}).copy()}

        return self._config

    def copy(self) -> "ControlMessage":
        config = self._config.copy()
        config["type"] = self.task_type().name

        tasks = []
        for (task_type, task_queue) in self.get_tasks().items():
            for task in task_queue:
                tasks.append({"type": task_type, "properties": task})

        config["tasks"] = tasks

        new_cm = ControlMessage(config)
        new_cm._payload = self._payload
        new_cm._tensors = self._tensors
        new_cm._timestamps = self._timestamps.copy()

        return new_cm

    def has_task(self, task_type: str) -> bool:
        """
        Return True if the control message has at least one task of the given type
        """
        # Using `get` to avoid creating an empty list if the task type is not present
        tasks = self._tasks.get(task_type, [])
        return len(tasks) > 0

    def add_task(self, task_type: str, properties: dict):
        if isinstance(task_type, str):
            cm_type = get_enum_members(ControlMessageType).get(task_type, ControlMessageType.NONE)
            if cm_type != ControlMessageType.NONE:
                if self._type == ControlMessageType.NONE:
                    self._type = cm_type
                elif self._type != cm_type:
                    raise ValueError("Cannot mix different types of tasks on the same control message")

        self._tasks[task_type].append(properties)

    def remove_task(self, task_type: str) -> dict:
        tasks = self._tasks.get(task_type, [])
        if len(tasks) == 0:
            raise ValueError(f"No task of type {task_type} found")

        return tasks.popleft()

    def get_tasks(self) -> dict[str, deque]:
        return self._tasks

    def set_metadata(self, key: str, value: typing.Any):
        self._config["metadata"][key] = value

    def has_metadata(self, key: str) -> bool:
        return key in self._config["metadata"]

    def get_metadata(self, key: str = None, default_value: typing.Any = None) -> typing.Any:
        """
        Return a given piece of metadata, if `key` is `None` return the entire metadata dictionary.
        If `key` is not found, `default_value` is returned.

        :param key: The key of the metadata to retrieve, or None for all metadata
        :param default_value: The value to return if the key is not found, ignored if `key` is None
        :return: The value of the metadata key, or the entire metadata dictionary if `key` is None
        """

        # Not using `get` since `None` is a valid value
        if key is None:
            return self._config["metadata"]

        return self._config["metadata"].get(key, default_value)

    def list_metadata(self) -> list[str]:
        return sorted(self._config["metadata"].keys())

    def payload(self, payload: MessageMeta = None) -> MessageMeta | None:
        if payload is not None:
            self._payload = payload

        return self._payload

    def tensors(self, tensors: TensorMemory = None) -> TensorMemory | None:
        if tensors is not None:
            self._tensors = tensors

        return self._tensors

    def task_type(self, new_task_type: ControlMessageType = None) -> ControlMessageType:
        if new_task_type is not None:
            self._type = new_task_type

        return self._type

    def set_timestamp(self, key: str, timestamp: datetime):
        self._timestamps[key] = timestamp

    def get_timestamp(self, key: str, fail_if_nonexist: bool = False) -> datetime | None:
        try:
            return self._timestamps[key]
        except KeyError:
            if fail_if_nonexist:
                raise
            return None

    def filter_timestamp(self, regex_filter: str) -> dict[str, datetime]:
        re_obj = re.compile(regex_filter)

        return {key: value for key, value in self._timestamps.items() if re_obj.match(key)}
