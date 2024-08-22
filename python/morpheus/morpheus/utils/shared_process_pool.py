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
from threading import Lock

# pylint: disable=W0201

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SerializableFuture:

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
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * 0.5))
                cls._instance._initialize(max_workers)
                logger.debug("SharedProcessPool has been initialized with %s workers.", max_workers)

            else:
                logger.debug("SharedProcessPool instance already exists.")

        return cls._instance

    def _initialize(self, total_max_workers):
        """
        Initialize a concurrent.futures.ProcessPoolExecutor instance.
        """
        self._total_max_workers = total_max_workers
        self._context = mp.get_context("fork")
        self._manager = self._context.Manager()
        self._total_usage = 0.0
        self._stage_usage = {}
        self._task_queues = self._manager.dict()
        self._stage_semaphores = self._manager.dict()
        self._processes = []

        # TODO: Test the performance of reading the shared variable in each worker loop and try some alternatives
        self._shutdown_in_progress = self._manager.Value("b", False)

        for i in range(total_max_workers):
            process = self._context.Process(target=self._worker,
                                            args=(self._task_queues, self._stage_semaphores,
                                                  self._shutdown_in_progress))
            process.start()
            self._processes.append(process)
            logger.debug("Process %s/%s has been started.", i + 1, total_max_workers)

    @staticmethod
    def _worker(task_queues, stage_semaphores, shutdown_in_progress):
        logger.debug("Worker process %s has been started.", os.getpid())

        while True:
            if shutdown_in_progress.value:
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

                # if task is None:  # Stop signal
                #     semaphore.release()
                #     return

                process_fn, args, future = task
                try:
                    result = process_fn(*args)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

                semaphore.release()

                time.sleep(0.1)  # Avoid busy-waiting

    def submit_task(self, stage_name, process_fn, *args):
        """
        Submit a task to the corresponding task queue of the stage.
        """
        future = SerializableFuture(self._context.Manager())
        task = (process_fn, args, future)
        self._task_queues[stage_name].put(task)

        return future

    def set_usage(self, stage_name, percentage):
        """
        Set the maximum percentage of processes that can be used by each stage.
        """
        if not 0 < percentage <= 1:
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
        if not self._shutdown:
            self._shutdown_in_progress.value = True
            # for stage_name, task_queue in self._task_queues.items():
            #     for _ in range(self._total_max_workers):
            #         task_queue.put(None)
            #     logger.debug("Task queue for stage %s has been cleared.", stage_name)

            for i, p in enumerate(self._processes):
                p.join()
                logger.debug("Process %s/%s has been terminated.", i + 1, self._total_max_workers)

            self._manager.shutdown()
            self._shutdown = True
            logger.debug("Process pool has been terminated.")

    def __del__(self):
        if not self._shutdown:
            self.shutdown()
