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
import multiprocessing as mp
import time

import numpy as np

from morpheus.utils.shared_process_pool import SharedProcessPool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _matrix_multiplication_task(size):
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)

    mul = np.dot(matrix_a, matrix_b)
    result = (mul, time.time())
    return result


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


def test_singleton():
    pool_1 = SharedProcessPool()
    pool_2 = SharedProcessPool()

    assert pool_1 is pool_2


def test_shared_process_pool():
    pool = SharedProcessPool()

    pool.set_usage("test_stage_1", 0.1)
    pool.set_usage("test_stage_2", 0.3)
    pool.set_usage("test_stage_3", 0.6)

    tasks = [("test_stage_1", 8000, 30), ("test_stage_2", 8000, 30), ("test_stage_3", 8000, 30)]

    processes = []
    for task in tasks:
        stage_name, task_size, num_tasks = task
        p = mp.Process(target=_test_worker, args=(pool, stage_name, task_size, num_tasks))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    test_shared_process_pool()
