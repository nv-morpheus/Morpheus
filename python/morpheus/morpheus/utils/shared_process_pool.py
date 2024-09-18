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
import queue
import time
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


class PoolStatus(Enum):
    INITIALIZING = 0
    RUNNING = 1
    STOPPING = 2
    SHUTDOWN = 3


class SimpleFuture:

    def __init__(self, manager):
        self._result = manager.Value("i", None)
        self._exception = manager.Value("i", None)
        self._done = manager.Event()

    def set_result(self, result):
        self._result.value = result
        self._done.set()

    def set_exception(self, exception):
        self._exception.value = exception
        self._done.set()

    def result(self):
        self._done.wait()
        if self._exception.value is not None:
            raise self._exception.value
        return self._result.value

    def done(self):
        return self._done.is_set()


class Task(SimpleFuture):

    def __init__(self, manager, process_fn, args, kwargs):
        super().__init__(manager)
        self._process_fn = process_fn
        self._args = args
        self._kwargs = kwargs

    @property
    def process_fn(self):
        return self._process_fn

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class CancellationToken:

    def __init__(self, manager):
        self._cancel_event = manager.Event()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self):
        return self._cancel_event.is_set()


# pylint: disable=W0201
class SharedProcessPool:

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Singleton pattern for SharedProcessPool.

        Returns
        -------
        cls._instance : SharedProcessPool
            The SharedProcessPool instance.

        Raises
        ------
        RuntimeError
            If SharedProcessPool() is called when the instance already exists but not running
        """

        with cls._lock:
            if cls._instance is None:
                logger.info("SharedProcessPool.__new__: Creating a new instance...")
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
                logger.info("SharedProcessPool has been initialized.")

            else:
                if cls._instance.status is not PoolStatus.RUNNING:
                    raise RuntimeError(
                        "SharedProcessPool instance already exists but it is not running. Please use start() or reset() to launch the pool."
                    )
                else:
                    logger.debug("SharedProcessPool.__new__: instance already exists and is currently running.")

        return cls._instance

    def _initialize(self):
        self._status = PoolStatus.INITIALIZING

        self._total_max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * 0.5))
        self._processes = []

        self._context = mp.get_context("fork")
        self._manager = self._context.Manager()
        self._task_queues = self._manager.dict()
        self._stage_semaphores = self._manager.dict()
        self._total_usage = 0.0
        self._stage_usage = {}

        self._cancellation_token = CancellationToken(self._manager)
        self._launch_workers()

        self._status = PoolStatus.RUNNING

    def _launch_workers(self):
        for i in range(self.total_max_workers):
            process = self._context.Process(target=self._worker,
                                            args=(self._cancellation_token, self._task_queues, self._stage_semaphores))
            process.start()
            self._processes.append(process)
            logger.debug("SharedProcessPool._lanch_workers(): Process %s/%s has been started.",
                         i + 1,
                         self.total_max_workers)

    @property
    def total_max_workers(self):
        return self._total_max_workers

    @property
    def status(self) -> PoolStatus:
        return self._status

    @staticmethod
    def _worker(cancellation_token, task_queues, stage_semaphores):
        logger.debug("SharedProcessPool._worker: Worker process %s has been started.", os.getpid())

        while True:
            if cancellation_token.is_cancelled():
                logger.debug("SharedProcessPool._worker: Worker process %s has terminated the worker loop.",
                             os.getpid())
                return

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

                process_fn = task.process_fn
                args = task.args
                kwargs = task.kwargs

                try:
                    result = process_fn(*args, **kwargs)
                    task.set_result(result)
                except Exception as e:
                    task.set_exception(e)

                semaphore.release()

                time.sleep(0.1)  # Avoid busy-waiting

    def submit_task(self, stage_name, process_fn, *args, **kwargs) -> Task:
        """
        Submit a task to the SharedProcessPool.

        Parameters
        ----------
        stage_name : str
            The unique name of the stage.
        process_fn : Callable
            The function to be executed in the process pool.
        args : Any
            Arbitrary arguments for the process_fn.
        kwargs : Any
            Arbitrary keyword arguments for the process_fn.

        Returns
        -------
        Task
            The task object that includes the result of the process_fn.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not running.
        ValueError
            If the stage_name has not been set in the SharedProcessPool.
        """
        if self._status != PoolStatus.RUNNING:
            raise RuntimeError("Cannot submit a task to a SharedProcessPool that is not running.")

        if stage_name not in self._stage_usage:
            raise ValueError(f"Stage {stage_name} has not been set in SharedProcessPool.")

        task = Task(self._manager, process_fn, args, kwargs)
        self._task_queues[stage_name].put(task)

        return task

    def set_usage(self, stage_name, percentage):
        """
        Set the usage of the SharedProcessPool for a specific stage.

        Parameters
        ----------
        stage_name : str
            The unique name of the stage.
        percentage : float
            The percentage of the total workers that will be allocated to the stage, should be between 0 and 1.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not running.
        ValueError
            If the percentage is not between 0 and 1 or the total usage is greater than 1.
        """
        if self._status != PoolStatus.RUNNING:
            raise RuntimeError("Cannot set usage to a SharedProcessPool that is not running.")

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

        logger.debug("SharedProcessPool.set_usage(): stage_usage: %s", self._stage_usage)
        logger.debug("SharedProcessPool.set_usage(): stage semaphores: %s", allowed_processes_num)

    def start(self):
        """
        Start the SharedProcessPool that is currently stopped and keep the settings before last shutdown.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not shutdown.
        """
        if self._status != PoolStatus.SHUTDOWN:
            raise RuntimeError("Cannot start a SharedProcessPool that is not shutdown.")

        self._launch_workers()
        self._status = PoolStatus.RUNNING

    def reset(self):
        """
        Clear all the previous settings and restart the SharedProcessPool.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not already shutdown.
        """
        if self._status != PoolStatus.SHUTDOWN:
            raise RuntimeError(
                "SharedProcessPool.reset(): Cannot reset a SharedProcessPool that is not already shutdown.")

        self._initialize()

    def stop(self):
        """
        Complete existing tasks and stop the SharedProcessPool.
        """
        if self._status not in (PoolStatus.RUNNING, PoolStatus.INITIALIZING):
            logger.info("SharedProcessPool.stop(): Cannot stop a SharedProcessPool that is not running.")
            return

        # no new tasks will be accepted from this point
        self._status = PoolStatus.STOPPING

        # wait for all task queues to be empty
        task_queue_count = len(self._task_queues)
        empty_task_queues = set()
        while len(empty_task_queues) < task_queue_count:
            for stage_name, task_queue in self._task_queues.items():
                if task_queue.empty():
                    empty_task_queues.add(stage_name)

        self._cancellation_token.cancel()

        for i, p in enumerate(self._processes):
            p.join()
            logger.debug("Process %s/%s has been joined.", i + 1, self._total_max_workers)

        logger.debug("SharedProcessPool.stop(): All tasks have been completed. SharedProcessPool has been stopped.")
        self._status = PoolStatus.SHUTDOWN

    def terminate(self):
        """
        Terminate all processes and shutdown the SharedProcessPool immediately.
        """
        if self._status not in (PoolStatus.RUNNING, PoolStatus.INITIALIZING):
            logger.info("SharedProcessPool.terminate():Cannot terminate a SharedProcessPool that is not running.")
            return

        for i, p in enumerate(self._processes):
            p.terminate()
            logger.debug("Process %s/%s has been terminated.", i + 1, self._total_max_workers)

        logger.debug("SharedProcessPool.terminate(): SharedProcessPool has been terminated.")
        self._status = PoolStatus.SHUTDOWN

    def wait_until_ready(self, timeout=None):
        """
        Wait until the SharedProcessPool is running and ready to accept tasks.

        Parameters
        ----------
        timeout : _type_, optional
            timeout in seconds to wait for the SharedProcessPool to be ready, by default None.
            If None, it will wait indefinitely.

        Raises
        ------
        RuntimeError
            If the SharedProcessPool is not initializing or running.
        TimeoutError
            If has been waiting more than the timeout.
        """
        if self.status not in (PoolStatus.INITIALIZING, PoolStatus.RUNNING):
            raise RuntimeError("Cannot wait for a SharedProcessPool that is not initializing.")

        start_time = time.time()
        while self.status != PoolStatus.RUNNING:
            if timeout is not None and timeout > 0 and time.time() - start_time > timeout:
                raise TimeoutError("SharedProcessPool wait_until_ready has timed out.")
            time.sleep(0.1)

        logger.debug("SharedProcessPool.wait_until_ready(): SharedProcessPool is ready.")

    def join(self, timeout=None):
        """
        Wait until the SharedProcessPool is terminated.

        Parameters
        ----------
        timeout : _type_, optional
            timeout in seconds to wait for the SharedProcessPool to be terminated, by default None.
            If None, it will wait indefinitely.

        Raises
        ------
        TimeoutError
            If has been waiting more than the timeout.
        """
        start_time = time.time()

        while self._status != PoolStatus.SHUTDOWN:
            if timeout is not None and timeout > 0 and time.time() - start_time > timeout:
                raise TimeoutError("SharedProcessPool join has timed out.")
            time.sleep(0.1)

        logger.debug("SharedProcessPool.join(): SharedProcessPool has been joined.")

    def __del__(self):
        if self._status != PoolStatus.SHUTDOWN:
            self.stop()
