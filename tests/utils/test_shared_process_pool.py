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

import atexit
import logging
import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from morpheus.utils.shared_process_pool import SharedProcessPool

logger = logging.getLogger(__name__)


@pytest.fixture(name="shared_process_pool")
def shared_process_pool_fixture():
    pool = SharedProcessPool()
    _reset_shared_process_pool(pool)

    atexit.register(pool.shutdown)  # make sure to shutdown the pool before the test exits

    return pool


def _matrix_multiplication_task(size):
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)

    mul = np.dot(matrix_a, matrix_b)
    result = (mul, time.time())
    return result


def _simple_add_task(a, b):
    return a + b


def _process_func_with_exception():
    raise ValueError("Exception is raised in the process.")


def _unserializable_function():
    return threading.Lock()


def _arbitrary_function(*args, **kwargs):
    return args, kwargs


# SharedProcessPool is a singleton, need to reset before each test
def _reset_shared_process_pool(pool: SharedProcessPool):
    for stage in pool._stage_usage:
        pool._stage_usage[stage] = 0.0
        pool._stage_semaphores[stage] = pool._manager.Semaphore(1)
        pool._task_queues[stage] = pool._manager.Queue()

    pool._total_usage = 0.0


def _test_worker(pool, stage_name, task_size, num_tasks):
    future_list = []
    for i in range(num_tasks):
        future_list.append(pool.submit_task(stage_name, _matrix_multiplication_task, task_size))
        logging.info("Task %s/%s has been submitted to stage %s.", i + 1, num_tasks, stage_name)

    for future in future_list:
        future.result()
        logging.info("task number %s has been completed in stage: %s", future_list.index(future), stage_name)

    logging.info("All tasks in stage %s have been completed in %.2f seconds.",
                 stage_name, (future_list[-1].result()[1] - future_list[0].result()[1]))

    assert len(future_list) == num_tasks


def test_singleton():
    pool_1 = SharedProcessPool()
    pool_2 = SharedProcessPool()

    assert pool_1 is pool_2


def test_single_task(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage", 0.5)

    a = 10
    b = 20

    future = pool.submit_task("test_stage", _simple_add_task, a, b)
    assert future.result() == a + b

    future = pool.submit_task("test_stage", _simple_add_task, a=a, b=b)
    assert future.result() == a + b

    future = pool.submit_task("test_stage", _simple_add_task, a, b=b)
    assert future.result() == a + b


def test_multiple_tasks(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage", 0.5)

    num_tasks = 100
    futures = []
    for _ in range(num_tasks):
        futures.append(pool.submit_task("test_stage", _simple_add_task, 10, 20))

    for future in futures:
        assert future.result() == 30


def test_error_process_function(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage", 0.5)

    with pytest.raises(ValueError):
        future = pool.submit_task("test_stage", _process_func_with_exception)
        future.result()


def test_unserializable_function(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage", 0.5)

    with pytest.raises(TypeError):
        future = pool.submit_task("test_stage", _unserializable_function)
        future.result()


def test_unserializable_arg(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage", 0.5)

    with pytest.raises(TypeError):
        future = pool.submit_task("test_stage", _arbitrary_function, threading.Lock())
        future.result()


def test_multiple_stages(shared_process_pool):
    pool = shared_process_pool

    task_size = 3000
    task_num = 30
    tasks = [("test_stage_1", task_size, task_num), ("test_stage_2", task_size, task_num),
             ("test_stage_3", task_size, task_num)]

    processes = []
    for task in tasks:
        stage_name, task_size, num_tasks = task
        p = mp.Process(target=_test_worker, args=(pool, stage_name, task_size, num_tasks))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


def test_invalid_stage_usage(shared_process_pool):
    pool = shared_process_pool

    with pytest.raises(ValueError):
        pool.set_usage("test_stage", 1.1)

    with pytest.raises(ValueError):
        pool.set_usage("test_stage", -0.1)

    pool.set_usage("test_stage_1", 0.5)
    pool.set_usage("test_stage_2", 0.4)

    pool.set_usage("test_stage_1", 0.6)  # ok to update the usage of an existing stage

    with pytest.raises(ValueError):
        pool.set_usage("test_stage_1", 0.7)  # not ok to exceed the total usage limit after updating

    with pytest.raises(ValueError):
        pool.set_usage("test_stage_3", 0.1)


def test_task_completion_before_shutdown(shared_process_pool):
    pool = shared_process_pool

    pool.set_usage("test_stage_1", 0.1)
    pool.set_usage("test_stage_2", 0.3)
    pool.set_usage("test_stage_3", 0.6)

    task_size = 3000
    task_num = 30
    futures = []
    for _ in range(task_num):
        futures.append(pool.submit_task("test_stage_1", _matrix_multiplication_task, task_size))
        futures.append(pool.submit_task("test_stage_2", _matrix_multiplication_task, task_size))
        futures.append(pool.submit_task("test_stage_3", _matrix_multiplication_task, task_size))

    pool.shutdown()

    # all tasks should be completed before shutdown
    assert len(futures) == 3 * task_num
    for future in futures:
        assert future.done()
