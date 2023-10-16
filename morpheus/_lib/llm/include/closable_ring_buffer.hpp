/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/core/expected.hpp"
#include "mrc/coroutines/schedule_policy.hpp"
#include "mrc/coroutines/thread_local_context.hpp"
#include "mrc/coroutines/thread_pool.hpp"

#include <glog/logging.h>

#include <atomic>
#include <coroutine>
#include <mutex>
#include <optional>
#include <vector>

namespace mrc::coroutines {

enum class RingBufferOpStatus
{
    Success,
    Stopped,
};

/**
 * @tparam ElementT The type of element the ring buffer will store.  Note that this type should be
 *         cheap to move if possible as it is moved into and out of the buffer upon write and
 *         read operations.
 */
template <typename ElementT>
class ClosableRingBuffer
{
    using mutex_type = std::mutex;

  public:
    struct Options
    {
        // capacity of ring buffer
        std::size_t capacity{8};

        // when there is an awaiting reader, the active execution context of the next writer will resume the awaiting
        // reader, the schedule_policy_t dictates how that is accomplished.
        SchedulePolicy reader_policy{SchedulePolicy::Reschedule};

        // when there is an awaiting writer, the active execution context of the next reader will resume the awaiting
        // writer, the producder_policy_t dictates how that is accomplished.
        SchedulePolicy writer_policy{SchedulePolicy::Reschedule};

        // when there is an awaiting writer, the active execution context of the next reader will resume the awaiting
        // writer, the producder_policy_t dictates how that is accomplished.
        SchedulePolicy completed_policy{SchedulePolicy::Reschedule};
    };

    /**
     * @throws std::runtime_error If `num_elements` == 0.
     */
    explicit ClosableRingBuffer(Options opts = {}) :
      m_elements(opts.capacity),  // elements needs to be extended from just holding ElementT to include a TraceContext
      m_num_elements(opts.capacity),
      m_writer_policy(opts.writer_policy),
      m_reader_policy(opts.reader_policy),
      m_completed_policy(opts.completed_policy)
    {
        if (m_num_elements == 0)
        {
            throw std::runtime_error{"num_elements cannot be zero"};
        }
    }

    ~ClosableRingBuffer()
    {
        // Wake up anyone still using the ring buffer.
        notify_waiters();
    }

    ClosableRingBuffer(const ClosableRingBuffer<ElementT>&) = delete;
    ClosableRingBuffer(ClosableRingBuffer<ElementT>&&)      = delete;

    auto operator=(const ClosableRingBuffer<ElementT>&) noexcept -> ClosableRingBuffer<ElementT>& = delete;
    auto operator=(ClosableRingBuffer<ElementT>&&) noexcept -> ClosableRingBuffer<ElementT>&      = delete;

    struct Operation
    {
        virtual void resume() = 0;
    };

    struct WriteOperation : ThreadLocalContext, Operation
    {
        WriteOperation(ClosableRingBuffer<ElementT>& rb, ElementT e) :
          m_rb(rb),
          m_e(std::move(e)),
          m_policy(m_rb.m_writer_policy)
        {}

        auto await_ready() noexcept -> bool
        {
            // return immediate if the buffer is closed
            if (m_rb.m_stopped.load(std::memory_order::acquire))
            {
                m_stopped = true;
                return true;
            }

            // start a span to time the write - this would include time suspended if the buffer is full
            // m_write_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});

            // the lock is owned by the operation, not scoped to the await_ready function
            m_lock = std::unique_lock(m_rb.m_mutex);
            return m_rb.try_write_locked(m_lock, m_e);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            auto lock = std::move(m_lock);  // use raii

            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine = awaiting_coroutine;
            m_next               = m_rb.m_write_waiters;
            m_rb.m_write_waiters = this;
            return true;
        }

        /**
         * @return write_result
         */
        auto await_resume() -> RingBufferOpStatus
        {
            ThreadLocalContext::resume_thread_local_context();
            return (!m_stopped ? RingBufferOpStatus::Success : RingBufferOpStatus::Stopped);
        }

        WriteOperation& use_scheduling_policy(SchedulePolicy policy) &
        {
            m_policy = policy;
            return *this;
        }

        WriteOperation use_scheduling_policy(SchedulePolicy policy) &&
        {
            m_policy = policy;
            return std::move(*this);
        }

        WriteOperation& resume_immediately() &
        {
            m_policy = SchedulePolicy::Immediate;
            return *this;
        }

        WriteOperation resume_immediately() &&
        {
            m_policy = SchedulePolicy::Immediate;
            return std::move(*this);
        }

        WriteOperation& resume_on(ThreadPool* thread_pool) &
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return *this;
        }

        WriteOperation resume_on(ThreadPool* thread_pool) &&
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return std::move(*this);
        }

      private:
        friend ClosableRingBuffer;

        void resume()
        {
            if (m_policy == SchedulePolicy::Immediate)
            {
                set_resume_on_thread_pool(nullptr);
            }
            resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        std::unique_lock<mutex_type> m_lock;
        /// The ring buffer the element is being written into.
        ClosableRingBuffer<ElementT>& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be written.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of write operations that are awaiting to write their element.
        WriteOperation* m_next{nullptr};
        /// The element this write operation is producing into the ring buffer.
        ElementT m_e;
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the ClosableRingBuffer, but can be overrided owner of the Awaiter
        SchedulePolicy m_policy;
        /// Span to measure the duration the writer spent writting data
        // trace::Handle<trace::Span> m_write_span{nullptr};
    };

    struct ReadOperation : ThreadLocalContext, Operation
    {
        explicit ReadOperation(ClosableRingBuffer<ElementT>& rb) : m_rb(rb), m_policy(m_rb.m_reader_policy) {}

        auto await_ready() noexcept -> bool
        {
            // the lock is owned by the operation, not scoped to the await_ready function
            m_lock = std::unique_lock(m_rb.m_mutex);
            // m_read_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});
            return m_rb.try_read_locked(m_lock, this);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            auto lock = std::move(m_lock);

            // the buffer is empty; don't suspend if the stop signal has been set.
            if (m_rb.m_stopped.load(std::memory_order::acquire))
            {
                m_stopped = true;
                return false;
            }

            // m_read_span->AddEvent("buffer_empty");
            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine = awaiting_coroutine;
            m_next               = m_rb.m_read_waiters;
            m_rb.m_read_waiters  = this;
            return true;
        }

        /**
         * @return The consumed element or std::nullopt if the read has failed.
         */
        auto await_resume() -> mrc::expected<ElementT, RingBufferOpStatus>
        {
            ThreadLocalContext::resume_thread_local_context();

            if (m_stopped)
            {
                return mrc::unexpected<RingBufferOpStatus>(RingBufferOpStatus::Stopped);
            }

            return std::move(m_e);
        }

        ReadOperation& use_scheduling_policy(SchedulePolicy policy)
        {
            m_policy = policy;
            return *this;
        }

        ReadOperation& resume_immediately()
        {
            m_policy = SchedulePolicy::Immediate;
            return *this;
        }

        ReadOperation& resume_on(ThreadPool* thread_pool)
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return *this;
        }

      private:
        friend ClosableRingBuffer;

        void resume()
        {
            if (m_policy == SchedulePolicy::Immediate)
            {
                set_resume_on_thread_pool(nullptr);
            }
            resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        std::unique_lock<mutex_type> m_lock;
        /// The ring buffer to read an element from.
        ClosableRingBuffer<ElementT>& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be consumed.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of read operations that are awaiting to read an element.
        ReadOperation* m_next{nullptr};
        /// The element this read operation will read.
        ElementT m_e;
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the ClosableRingBuffer, but can be overrided owner of the Awaiter
        SchedulePolicy m_policy;
        /// Span measure time awaiting on reading data
        // trace::Handle<trace::Span> m_read_span;
    };

    struct CompletedOperation : ThreadLocalContext, Operation
    {
        explicit CompletedOperation(ClosableRingBuffer<ElementT>& rb) : m_rb(rb), m_policy(m_rb.m_completed_policy) {}

        auto await_ready() noexcept -> bool
        {
            // the lock is owned by the operation, not scoped to the await_ready function
            m_lock = std::unique_lock(m_rb.m_mutex);
            // m_read_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});
            return m_rb.try_completed_locked(m_lock, this);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            auto lock = std::move(m_lock);

            // m_read_span->AddEvent("buffer_empty");
            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine     = awaiting_coroutine;
            m_next                   = m_rb.m_completed_waiters;
            m_rb.m_completed_waiters = this;
            return true;
        }

        /**
         * @return The consumed element or std::nullopt if the read has failed.
         */
        auto await_resume()
        {
            ThreadLocalContext::resume_thread_local_context();
        }

        ReadOperation& use_scheduling_policy(SchedulePolicy policy)
        {
            m_policy = policy;
            return *this;
        }

        ReadOperation& resume_immediately()
        {
            m_policy = SchedulePolicy::Immediate;
            return *this;
        }

        ReadOperation& resume_on(ThreadPool* thread_pool)
        {
            m_policy = SchedulePolicy::Reschedule;
            set_resume_on_thread_pool(thread_pool);
            return *this;
        }

      private:
        friend ClosableRingBuffer;

        void resume()
        {
            if (m_policy == SchedulePolicy::Immediate)
            {
                set_resume_on_thread_pool(nullptr);
            }
            resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        std::unique_lock<mutex_type> m_lock;
        /// The ring buffer to read an element from.
        ClosableRingBuffer<ElementT>& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be consumed.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of read operations that are awaiting to read an element.
        CompletedOperation* m_next{nullptr};
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the ClosableRingBuffer, but can be overrided owner of the Awaiter
        SchedulePolicy m_policy;
        /// Span measure time awaiting on reading data
        // trace::Handle<trace::Span> m_read_span;
    };

    /**
     * Produces the given element into the ring buffer.  This operation will suspend until a slot
     * in the ring buffer becomes available.
     * @param e The element to write.
     */
    [[nodiscard]] auto write(ElementT e) -> WriteOperation
    {
        return WriteOperation{*this, std::move(e)};
    }

    /**
     * Consumes an element from the ring buffer.  This operation will suspend until an element in
     * the ring buffer becomes available.
     */
    [[nodiscard]] auto read() -> ReadOperation
    {
        return ReadOperation{*this};
    }

    /**
     * Blocks until `close()` has been called and all elements have been returned
     */
    [[nodiscard]] auto completed() -> CompletedOperation
    {
        return CompletedOperation{*this};
    }

    void close()
    {
        // if there are awaiting readers, then we must wait them up and signal that the buffer is closed;
        // otherwise, mark the buffer as closed and fail all new writes immediately. readers should be allowed
        // to keep reading until the buffer is empty. when the buffer is empty, readers will fail to suspend and exit
        // with a stopped status

        // Only wake up waiters once.
        if (m_stopped.load(std::memory_order::acquire))
        {
            return;
        }

        std::unique_lock lk{m_mutex};
        m_stopped.exchange(true, std::memory_order::release);

        // the buffer is empty and no more items will be added
        if (m_used == 0)
        {
            // there should be no awaiting writers
            CHECK(m_write_waiters == nullptr);

            // signal all awaiting readers that the buffer is stopped
            while (m_read_waiters != nullptr)
            {
                auto* to_resume      = m_read_waiters;
                to_resume->m_stopped = true;
                m_read_waiters       = m_read_waiters->m_next;

                lk.unlock();
                to_resume->resume();
                lk.lock();
            }

            // signal all awaiting completed that the buffer is completed
            while (m_completed_waiters != nullptr)
            {
                auto* to_resume      = m_completed_waiters;
                to_resume->m_stopped = true;
                m_completed_waiters  = m_completed_waiters->m_next;

                lk.unlock();
                to_resume->resume();
                lk.lock();
            }
        }
    }

    bool is_closed() const noexcept
    {
        return m_stopped.load(std::memory_order::acquire);
    }

    /**
     * @return The current number of elements contained in the ring buffer.
     */
    auto size() const -> size_t
    {
        std::atomic_thread_fence(std::memory_order::acquire);
        return m_used;
    }

    /**
     * @return True if the ring buffer contains zero elements.
     */
    auto empty() const -> bool
    {
        return size() == 0;
    }

    /**
     * Wakes up all currently awaiting writers and readers.  Their await_resume() function
     * will return an expected read result that the ring buffer has stopped.
     */
    auto notify_waiters() -> void
    {
        // Only wake up waiters once.
        if (m_stopped.load(std::memory_order::acquire))
        {
            return;
        }

        std::unique_lock lk{m_mutex};
        m_stopped.exchange(true, std::memory_order::release);

        while (m_write_waiters != nullptr)
        {
            auto* to_resume      = m_write_waiters;
            to_resume->m_stopped = true;
            m_write_waiters      = m_write_waiters->m_next;

            lk.unlock();
            to_resume->resume();
            lk.lock();
        }

        while (m_read_waiters != nullptr)
        {
            auto* to_resume      = m_read_waiters;
            to_resume->m_stopped = true;
            m_read_waiters       = m_read_waiters->m_next;

            lk.unlock();
            to_resume->resume();
            lk.lock();
        }

        while (m_completed_waiters != nullptr)
        {
            auto* to_resume      = m_completed_waiters;
            to_resume->m_stopped = true;
            m_completed_waiters  = m_completed_waiters->m_next;

            lk.unlock();
            to_resume->resume();
            lk.lock();
        }
    }

  private:
    friend WriteOperation;
    friend ReadOperation;
    friend CompletedOperation;

    mutex_type m_mutex{};

    std::vector<ElementT> m_elements;
    const std::size_t m_num_elements;
    const SchedulePolicy m_writer_policy;
    const SchedulePolicy m_reader_policy;
    const SchedulePolicy m_completed_policy;

    /// The current front pointer to an open slot if not full.
    size_t m_front{0};
    /// The current back pointer to the oldest item in the buffer if not empty.
    size_t m_back{0};
    /// The number of items in the ring buffer.
    size_t m_used{0};

    /// The LIFO list of write waiters - single writers will have order perserved
    //  Note: if there are multiple writers order can not be guaranteed, so no need for FIFO
    WriteOperation* m_write_waiters{nullptr};
    /// The LIFO list of read watier.
    ReadOperation* m_read_waiters{nullptr};
    /// The LIFO list of completed watier.
    CompletedOperation* m_completed_waiters{nullptr};

    std::atomic<bool> m_stopped{false};

    auto try_write_locked(std::unique_lock<mutex_type>& lk, ElementT& e) -> bool
    {
        if (m_used == m_num_elements)
        {
            DCHECK(m_read_waiters == nullptr);
            return false;
        }

        // We will be able to write an element into the buffer.
        m_elements[m_front] = std::move(e);
        m_front             = (m_front + 1) % m_num_elements;
        ++m_used;

        ReadOperation* to_resume = nullptr;

        if (m_read_waiters != nullptr)
        {
            to_resume      = m_read_waiters;
            m_read_waiters = m_read_waiters->m_next;

            // Since the read operation suspended it needs to be provided an element to read.
            to_resume->m_e = std::move(m_elements[m_back]);
            m_back         = (m_back + 1) % m_num_elements;
            --m_used;  // And we just consumed up another item.
        }

        // After this point we will no longer be checking state objects on the buffer
        lk.unlock();

        if (to_resume != nullptr)
        {
            to_resume->resume();
        }

        return true;
    }

    auto try_read_locked(std::unique_lock<mutex_type>& lk, ReadOperation* op) -> bool
    {
        if (m_used == 0)
        {
            return false;
        }

        // We will be successful in reading an element from the buffer.
        op->m_e = std::move(m_elements[m_back]);
        m_back  = (m_back + 1) % m_num_elements;
        --m_used;

        WriteOperation* writer_to_resume = nullptr;

        if (m_write_waiters != nullptr)
        {
            writer_to_resume = m_write_waiters;
            m_write_waiters  = m_write_waiters->m_next;

            // Since the write operation suspended it needs to be provided a slot to place its element.
            m_elements[m_front] = std::move(writer_to_resume->m_e);
            m_front             = (m_front + 1) % m_num_elements;
            ++m_used;  // And we just written another item.
        }

        CompletedOperation* completed_waiters = nullptr;

        // Check if we are stopped and there are no more elements in the buffer.
        if (m_used == 0 && m_stopped.load(std::memory_order::acquire))
        {
            completed_waiters   = m_completed_waiters;
            m_completed_waiters = nullptr;
        }

        // After this point we will no longer be checking state objects on the buffer
        lk.unlock();

        // Resume any writer
        if (writer_to_resume != nullptr)
        {
            DCHECK(completed_waiters == nullptr) << "Logic error. Wrote value but count is 0";

            writer_to_resume->resume();
        }

        // Resume completed if there are any
        while (completed_waiters != nullptr)
        {
            completed_waiters->resume();

            completed_waiters = completed_waiters->m_next;
        }

        return true;
    }

    auto try_completed_locked(std::unique_lock<mutex_type>& lk, CompletedOperation* op) -> bool
    {
        // Condition is already met, no need to wait
        if (!m_stopped.load(std::memory_order::acquire) || m_used >= 0)
        {
            return false;
        }

        DCHECK(m_write_waiters == nullptr) << "Should not have any writers with a closed buffer";

        // release lock
        lk.unlock();

        return true;
    }
};

}  // namespace mrc::coroutines
