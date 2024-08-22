import logging
import atexit
import numpy as np
import multiprocessing as mp
import time
from morpheus.utils.shared_process_pool import SharedProcessPool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _matrix_multiplication_task(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    mul = np.dot(A, B)
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


def test_shared_process_pool():
    pool = SharedProcessPool()
    atexit.register(pool.shutdown)

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
