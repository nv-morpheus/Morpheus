# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import math
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import typing
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


class PoolStatus(Enum):
    INITIALIZING = 0
    RUNNING = 1
    STOPPING = 2
    SHUTDOWN = 3


@dataclass
class Task:
    id: uuid.UUID
    process_fn: typing.Callable
    args: tuple
    kwargs: dict


@dataclass
class TaskResult:
    id: uuid.UUID
    result: typing.Any
    exception: typing.Optional[Exception] = None


# pylint: disable=W0201
class SharedProcessPool:

    _instance = None
    _lock = Lock()
    _shutdown = False

    def __new__(cls):
        """
            Initialize as a singleton.
        """
        logger.debug("Creating a new instance of SharedProcessPool...")
        with cls._lock:

            max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * 0.5))

            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(max_workers)
                logger.debug("SharedProcessPool has been initialized with %s workers.", max_workers)

            else:
                if cls._instance.status is not PoolStatus.RUNNING:
                    cls._instance._initialize(max_workers)
                    logger.debug("SharedProcessPool has been initialized with %s workers.", max_workers)
                else:
                    logger.debug("SharedProcessPool instance already exists and currently running.")

        return cls._instance

    def _initialize(self, total_max_workers):
        """
        Initialize a concurrent.futures.ProcessPoolExecutor instance.
        """
        self._status = PoolStatus.INITIALIZING

        self._total_max_workers = total_max_workers
        self._processes = []
        self._total_usage = 0.0
        self._stage_usage = {}

        self._context = mp.get_context("fork")
        self._manager = self._context.Manager()

        self._task_queues = self._manager.dict()
        self._task_futures: typing.Dict[uuid.UUID, Future] = {}
        self._stage_semaphores = self._manager.dict()

        self._completion_queue = self._manager.Queue()

        self._shutdown_flag = self._manager.Value("b", False)
        self._shutdown_event = threading.Event()

        for i in range(total_max_workers):
            process = self._context.Process(target=self._worker,
                                            args=(self._task_queues,
                                                  self._stage_semaphores,
                                                  self._completion_queue,
                                                  self._shutdown_flag))
            process.start()
            self._processes.append(process)
            logger.debug("Process %s/%s has been started.", i + 1, total_max_workers)

        self._task_result_collection_thread = threading.Thread(target=self._task_result_collection_loop,
                                                               args=(self._completion_queue,
                                                                     self._task_futures,
                                                                     self._shutdown_event))
        self._task_result_collection_thread.start()
        self._status = PoolStatus.RUNNING

    @property
    def total_max_workers(self):
        return self._total_max_workers

    @property
    def status(self) -> PoolStatus:
        return self._status

    @staticmethod
    def _worker(task_queues, stage_semaphores, completion_queue, shutdown_flag):
        logger.debug("Worker process %s has been started.", os.getpid())

        while True:
            if shutdown_flag.value:
                logger.debug("Worker process %s has been terminated.", os.getpid())
                return

            # iterate over every semaphore
            for stage_name, task_queue in task_queues.items():
                semaphore = stage_semaphores[stage_name]

                if not semaphore.acquire(blocking=False):
                    # Stage has reached the limitation of processes
                    continue

                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    semaphore.release()
                    continue

                if task is None:
                    logger.warning("Worker process %s received a None task.", os.getpid())
                    semaphore.release()
                    continue

                process_fn = task.process_fn
                args = task.args
                kwargs = task.kwargs

                task_result = TaskResult(task.id, None, None)
                try:
                    result = process_fn(*args, **kwargs)
                    task_result.result = result
                except Exception as e:
                    task_result.exception = e

                try:
                    completion_queue.put(task_result)
                # the result must be serializable
                except (pickle.PicklingError, TypeError) as e:
                    task_result.exception = e
                    task_result.result = None
                    completion_queue.put(task_result)

                semaphore.release()

                time.sleep(0.1)  # Avoid busy-waiting

    @staticmethod
    def _task_result_collection_loop(completion_queue, task_futures, shutdown_event):
        while True:
            if shutdown_event.is_set() and completion_queue.empty():
                logger.debug("Task result collection process has been terminated.")
                return
            try:
                task_result = completion_queue.get_nowait()

                task_id = task_result.id
                future = task_futures.pop(task_id)

                if task_result.exception is not None:
                    future.set_exception(task_result.exception)
                else:
                    future.set_result(task_result.result)

            except queue.Empty:
                time.sleep(0.1)

    def submit_task(self, stage_name, process_fn, *args, **kwargs) -> Future:
        """
        Submit a task to the corresponding task queue of the stage.
        """
        with self._lock:
            task = Task(uuid.uuid4(), process_fn, args, kwargs)
            future = Future()
            self._task_futures[task.id] = future
            self._task_queues[stage_name].put(task)

            return future

    def set_usage(self, stage_name, percentage):
        """
        Set the maximum percentage of processes that can be used by each stage.
        """
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1.")

        new_total_usage = self._total_usage - self._stage_usage.get(stage_name, 0.0) + percentage

        if new_total_usage > 1.0:
            raise ValueError("Total percentage cannot exceed 1.0.")

        self._stage_usage[stage_name] = percentage
        self._total_usage = new_total_usage

        allowed_processes_num = max(1, int(self._total_max_workers * percentage))
        self._stage_semaphores[stage_name] = self._manager.Semaphore(allowed_processes_num)

        if stage_name not in self._task_queues:
            self._task_queues[stage_name] = self._manager.Queue()

        logger.debug("stage_usage: %s", self._stage_usage)
        logger.debug("stage semaphores: %s", allowed_processes_num)

    def shutdown(self):
        if self._status != PoolStatus.SHUTDOWN:

            self._status = PoolStatus.STOPPING

            while self._task_futures:
                time.sleep(0.1)

            self._shutdown_flag.value = True

            for i, p in enumerate(self._processes):
                p.join()
                logger.debug("Process %s/%s has been terminated.", i + 1, self._total_max_workers)

            self._shutdown_event.set()
            self._task_result_collection_thread.join()

            self._manager.shutdown()
            self._shutdown = True
            self._status = PoolStatus.SHUTDOWN
            logger.debug("Process pool has been terminated.")

    def __del__(self):
        if self._status != PoolStatus.SHUTDOWN:
            self.shutdown()
