/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Original Source: https://github.com/jbaldwin/libcoro
 * Original License: Apache License, Version 2.0; included below
 */

/**
 * Copyright 2021 Josh Baldwin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mrc/coroutines/concepts/executor.hpp"
#include "mrc/coroutines/scheduler.hpp"
#include "mrc/coroutines/task.hpp"

#include <atomic>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace mrc::coroutines {
class io_scheduler;

class TaskContainer
{
  public:
    using task_position_t = std::list<std::optional<Task<>>>::iterator;

    /**
     * @param e Tasks started in the container are scheduled onto this executor.  For tasks created
     *           from a coro::io_scheduler, this would usually be that coro::io_scheduler instance.
     * @param opts Task container options.
     */
    TaskContainer(std::shared_ptr<Scheduler> e) : m_scheduler(std::move(e))
    {
        if (m_scheduler == nullptr)
        {
            throw std::runtime_error{"task_container cannot have a nullptr executor"};
        }
    }
    TaskContainer(const TaskContainer&)                    = delete;
    TaskContainer(TaskContainer&&)                         = delete;
    auto operator=(const TaskContainer&) -> TaskContainer& = delete;
    auto operator=(TaskContainer&&) -> TaskContainer&      = delete;
    ~TaskContainer()
    {
        // This will hang the current thread.. but if tasks are not complete thats also pretty bad.
        while (!empty())
        {
            garbage_collect();
        }
    }

    enum class GarbageCollectPolicy
    {
        /// Execute garbage collection.
        yes,
        /// Do not execute garbage collection.
        no
    };

    /**
     * Stores a user task and starts its execution on the container's thread pool.
     * @param user_task The scheduled user's task to store in this task container and start its execution.
     * @param cleanup Should the task container run garbage collect at the beginning of this store
     *                call?  Calling at regular intervals will reduce memory usage of completed
     *                tasks and allow for the task container to re-use allocated space.
     */
    auto start(Task<void>&& user_task, GarbageCollectPolicy cleanup = GarbageCollectPolicy::yes) -> void
    {
        m_size.fetch_add(1, std::memory_order::relaxed);

        std::lock_guard lk{m_mutex};

        if (cleanup == GarbageCollectPolicy::yes)
        {
            gc_internal();
        }

        // Store the task inside a cleanup task for self deletion.
        auto pos  = m_tasks.emplace(m_tasks.end(), std::nullopt);
        auto task = make_cleanup_task(std::move(user_task), pos);
        *pos      = std::move(task);

        // Start executing from the cleanup task to schedule the user's task onto the thread pool.
        pos->value().resume();
    }

    /**
     * Garbage collects any tasks that are marked as deleted.  This frees up space to be re-used by
     * the task container for newly stored tasks.
     * @return The number of tasks that were deleted.
     */
    auto garbage_collect() -> std::size_t  // __attribute__((used))
    {
        std::lock_guard lk{m_mutex};
        return gc_internal();
    }

    /**
     * @return The number of tasks that are awaiting deletion.
     */
    auto delete_task_size() const -> std::size_t
    {
        std::atomic_thread_fence(std::memory_order::acquire);
        return m_tasks_to_delete.size();
    }

    /**
     * @return True if there are no tasks awaiting deletion.
     */
    auto delete_tasks_empty() const -> bool
    {
        std::atomic_thread_fence(std::memory_order::acquire);
        return m_tasks_to_delete.empty();
    }

    /**
     * @return The number of active tasks in the container.
     */
    auto size() const -> std::size_t
    {
        return m_size.load(std::memory_order::relaxed);
    }

    /**
     * @return True if there are no active tasks in the container.
     */
    auto empty() const -> bool
    {
        return size() == 0;
    }

    /**
     * Will continue to garbage collect and yield until all tasks are complete.  This method can be
     * co_await'ed to make it easier to wait for the task container to have all its tasks complete.
     *
     * This does not shut down the task container, but can be used when shutting down, or if your
     * logic requires all the tasks contained within to complete, it is similar to coro::latch.
     */
    auto garbage_collect_and_yield_until_empty() -> Task<void>
    {
        while (!empty())
        {
            garbage_collect();
            co_await m_scheduler->yield();
        }
    }

  private:
    /**
     * Interal GC call, expects the public function to lock.
     */
    auto gc_internal() -> std::size_t
    {
        std::size_t deleted{0};
        if (!m_tasks_to_delete.empty())
        {
            for (const auto& pos : m_tasks_to_delete)
            {
                if (pos->has_value())
                {
                    pos->value().destroy();
                }
                m_tasks.erase(pos);
            }
            deleted = m_tasks_to_delete.size();
            m_tasks_to_delete.clear();
        }
        return deleted;
    }

    /**
     * Encapsulate the users tasks in a cleanup task which marks itself for deletion upon
     * completion.  Simply co_await the users task until its completed and then mark the given
     * position within the task manager as being deletable.  The scheduler's next iteration
     * in its event loop will then free that position up to be re-used.
     *
     * This function will also unconditionally catch all unhandled exceptions by the user's
     * task to prevent the scheduler from throwing exceptions.
     * @param user_task The user's task.
     * @param pos The position where the task data will be stored in the task manager.
     * @return The user's task wrapped in a self cleanup task.
     */
    auto make_cleanup_task(Task<void> user_task, task_position_t pos) -> Task<void>
    {
        // Immediately move the task onto the executor.
        co_await m_scheduler->schedule();

        try
        {
            // Await the users task to complete.
            co_await user_task;
        } catch (const std::exception& e)
        {
            // what would be a good way to report this to the user...?  Catching here is required
            // since the co_await will unwrap the unhandled exception on the task.
            // The user's task should ideally be wrapped in a catch all and handle it themselves, but
            // that cannot be guaranteed.
            std::cerr << "Task_container user_task had an unhandled exception e.what()= " << e.what() << "\n";
        } catch (...)
        {
            // don't crash if they throw something that isn't derived from std::exception
            std::cerr << "Task_container user_task had unhandle exception, not derived from std::exception.\n";
        }

        std::lock_guard lk{m_mutex};
        m_tasks_to_delete.push_back(pos);
        m_size.fetch_sub(1, std::memory_order::relaxed);
        co_return;
    }

    /// Mutex for safely mutating the task containers across threads, expected usage is within
    /// thread pools for indeterminate lifetime requests.
    std::mutex m_mutex{};
    /// The number of alive tasks.
    std::atomic<std::size_t> m_size{};
    /// Maintains the lifetime of the tasks until they are completed.
    std::list<std::optional<Task<void>>> m_tasks{};
    /// The set of tasks that have completed and need to be deleted.
    std::vector<task_position_t> m_tasks_to_delete{};
    /// The current free position within the task indexes list.  Anything before
    std::shared_ptr<Scheduler> m_scheduler{nullptr};
};

}  // namespace mrc::coroutines
