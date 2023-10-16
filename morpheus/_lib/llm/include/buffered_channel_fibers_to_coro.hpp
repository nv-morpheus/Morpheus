
//          Copyright Oliver Kowalke 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_FIBERS_BUFFERED_CHANNEL_H
#define BOOST_FIBERS_BUFFERED_CHANNEL_H

#include <boost/config.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <boost/fiber/context.hpp>
#include <boost/fiber/detail/config.hpp>
#include <boost/fiber/detail/convert.hpp>
#include <boost/fiber/detail/spinlock.hpp>
#include <boost/fiber/exceptions.hpp>
#include <boost/fiber/waker.hpp>
#include <mrc/core/expected.hpp>
#include <mrc/coroutines/ring_buffer.hpp>
#include <mrc/coroutines/schedule_policy.hpp>
#include <mrc/coroutines/thread_local_context.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <type_traits>

#ifdef BOOST_HAS_ABI_HEADERS
    #include BOOST_ABI_PREFIX
#endif

namespace mrc::coroutines {

class IWakeable
{
  public:
    virtual void wake() = 0;
};

class Waker
{
  public:
    Waker(IWakeable* wakable) : m_wakable(wakable) {}

    bool wake()
    {
        m_wakable->wake();

        return true;
    }

    IWakeable* m_wakable{nullptr};
};

class WaitQueue
{
  public:
    void suspend_and_wait(boost::fibers::detail::spinlock_lock& lk, IWakeable* wakable)
    {
        m_waiters.emplace_back(wakable);

        lk.unlock();
    }

    void notify_one()
    {
        while (!m_waiters.empty())
        {
            auto& w = m_waiters.front();
            m_waiters.pop_front();

            if (w.wake())
            {
                break;
            }
        }
    }

    void notify_all()
    {
        while (!m_waiters.empty())
        {
            auto& w = m_waiters.front();
            m_waiters.pop_front();

            w.wake();
        }
    }

    bool empty() const
    {
        return m_waiters.empty();
    }

  private:
    std::list<Waker> m_waiters;
};

}  // namespace mrc::coroutines

namespace boost {
namespace fibers {

template <typename T>
class buffered_channel_fibers_to_coro
{
  public:
    using value_type = typename std::remove_reference<T>::type;

    struct ReadOperation : mrc::coroutines::ThreadLocalContext, mrc::coroutines::IWakeable
    {
        using parent_t = buffered_channel_fibers_to_coro<T>;

        explicit ReadOperation(parent_t& rb) : m_rb(rb), m_policy(m_rb.m_reader_policy) {}

        auto await_ready() noexcept -> bool
        {
            return false;
            // // the lock is owned by the operation, not scoped to the await_ready function
            // m_lock = std::unique_lock(m_rb.m_mutex);
            // // m_read_span->AddEvent("start_on", {{"thead.id", mrc::this_thread::get_id()}});
            // return m_rb.try_read_locked(m_lock, this);
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
        {
            // // m_lock was acquired as part of await_ready; await_suspend is responsible for releasing the lock
            // auto lock = std::move(m_lock);

            // // the buffer is empty; don't suspend if the stop signal has been set.
            // if (m_rb.m_stopped.load(std::memory_order::acquire))
            // {
            //     m_stopped = true;
            //     return false;
            // }

            // m_read_span->AddEvent("buffer_empty");
            ThreadLocalContext::suspend_thread_local_context();

            m_awaiting_coroutine = awaiting_coroutine;

            m_rb.m_read_waiters.suspend_and_wait(m_lock, this);

            // m_next              = m_rb.m_read_waiters;
            // m_rb.m_read_waiters = this;
            return true;
        }

        /**
         * @return The consumed element or std::nullopt if the read has failed.
         */
        auto await_resume() -> bool
        {
            ThreadLocalContext::resume_thread_local_context();

            return m_stopped;

            // if (m_stopped)
            // {
            //     return mrc::unexpected<mrc::coroutines::RingBufferOpStatus>(
            //         mrc::coroutines::RingBufferOpStatus::Stopped);
            // }

            // return std::move(m_e);
        }

        // ReadOperation& use_scheduling_policy(mrc::coroutines::SchedulePolicy policy)
        // {
        //     m_policy = policy;
        //     return *this;
        // }

        // ReadOperation& resume_immediately()
        // {
        //     m_policy = mrc::coroutines::SchedulePolicy::Immediate;
        //     return *this;
        // }

        // ReadOperation& resume_on(mrc::coroutines::ThreadPool* thread_pool)
        // {
        //     m_policy = mrc::coroutines::SchedulePolicy::Reschedule;
        //     this->set_resume_on_thread_pool(thread_pool);
        //     return *this;
        // }

      private:
        friend parent_t;

        void wake() override
        {
            // Reacquire the lock
            // m_lock.lock();

            if (m_policy == mrc::coroutines::SchedulePolicy::Immediate)
            {
                this->set_resume_on_thread_pool(nullptr);
            }
            this->resume_coroutine(m_awaiting_coroutine);
        }

        /// The lock is acquired in await_ready; if ready it is release; otherwise, await_suspend should release it
        detail::spinlock_lock& m_lock;
        /// The ring buffer to read an element from.
        parent_t& m_rb;
        /// If the operation needs to suspend, the coroutine to resume when the element can be consumed.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// Linked list of read operations that are awaiting to read an element.
        // ReadOperation* m_next{nullptr};
        /// The element this read operation will read.
        T m_e;
        /// Was the operation stopped?
        bool m_stopped{false};
        /// Scheduling Policy - default provided by the RingBuffer, but can be overrided owner of the Awaiter
        mrc::coroutines::SchedulePolicy m_policy;
        /// Span measure time awaiting on reading data
        // trace::Handle<trace::Span> m_read_span;
    };

  private:
    using slot_type = value_type;

    mutable detail::spinlock splk_{};
    wait_queue waiting_producers_{};
    // wait_queue                                          waiting_consumers_{};
    mrc::coroutines::WaitQueue waiting_consumers_;
    slot_type* slots_;
    std::size_t pidx_{0};
    std::size_t cidx_{0};
    std::size_t capacity_;
    bool closed_{false};

    bool is_full_() const noexcept
    {
        return cidx_ == ((pidx_ + 1) % capacity_);
    }

    bool is_empty_() const noexcept
    {
        return cidx_ == pidx_;
    }

    bool is_closed_() const noexcept
    {
        return closed_;
    }

  public:
    explicit buffered_channel_fibers_to_coro(std::size_t capacity) : capacity_{capacity}
    {
        if (BOOST_UNLIKELY(2 > capacity_ || 0 != (capacity_ & (capacity_ - 1))))
        {
            throw fiber_error{std::make_error_code(std::errc::invalid_argument),
                              "boost fiber: buffer capacity is invalid"};
        }
        slots_ = new slot_type[capacity_];
    }

    ~buffered_channel_fibers_to_coro()
    {
        close();
        delete[] slots_;
    }

    buffered_channel_fibers_to_coro(buffered_channel_fibers_to_coro const&)            = delete;
    buffered_channel_fibers_to_coro& operator=(buffered_channel_fibers_to_coro const&) = delete;

    bool is_closed() const noexcept
    {
        detail::spinlock_lock lk{splk_};
        return is_closed_();
    }

    void close() noexcept
    {
        detail::spinlock_lock lk{splk_};
        if (!closed_)
        {
            closed_ = true;
            waiting_producers_.notify_all();
            waiting_consumers_.notify_all();
        }
    }

    channel_op_status try_push(value_type const& value)
    {
        detail::spinlock_lock lk{splk_};
        if (BOOST_UNLIKELY(is_closed_()))
        {
            return channel_op_status::closed;
        }
        if (is_full_())
        {
            return channel_op_status::full;
        }
        slots_[pidx_] = value;
        pidx_         = (pidx_ + 1) % capacity_;
        waiting_consumers_.notify_one();
        return channel_op_status::success;
    }

    channel_op_status try_push(value_type&& value)
    {
        detail::spinlock_lock lk{splk_};
        if (BOOST_UNLIKELY(is_closed_()))
        {
            return channel_op_status::closed;
        }
        if (is_full_())
        {
            return channel_op_status::full;
        }
        slots_[pidx_] = std::move(value);
        pidx_         = (pidx_ + 1) % capacity_;
        waiting_consumers_.notify_one();
        return channel_op_status::success;
    }

    channel_op_status push(value_type const& value)
    {
        context* active_ctx = context::active();
        for (;;)
        {
            detail::spinlock_lock lk{splk_};
            if (BOOST_UNLIKELY(is_closed_()))
            {
                return channel_op_status::closed;
            }
            if (is_full_())
            {
                waiting_producers_.suspend_and_wait(lk, active_ctx);
            }
            else
            {
                slots_[pidx_] = value;
                pidx_         = (pidx_ + 1) % capacity_;
                waiting_consumers_.notify_one();
                return channel_op_status::success;
            }
        }
    }

    channel_op_status push(value_type&& value)
    {
        context* active_ctx = context::active();
        for (;;)
        {
            detail::spinlock_lock lk{splk_};
            if (BOOST_UNLIKELY(is_closed_()))
            {
                return channel_op_status::closed;
            }
            if (is_full_())
            {
                waiting_producers_.suspend_and_wait(lk, active_ctx);
            }
            else
            {
                slots_[pidx_] = std::move(value);
                pidx_         = (pidx_ + 1) % capacity_;

                waiting_consumers_.notify_one();
                return channel_op_status::success;
            }
        }
    }

    template <typename Rep, typename Period>
    channel_op_status push_wait_for(value_type const& value, std::chrono::duration<Rep, Period> const& timeout_duration)
    {
        return push_wait_until(value, std::chrono::steady_clock::now() + timeout_duration);
    }

    template <typename Rep, typename Period>
    channel_op_status push_wait_for(value_type&& value, std::chrono::duration<Rep, Period> const& timeout_duration)
    {
        return push_wait_until(std::forward<value_type>(value), std::chrono::steady_clock::now() + timeout_duration);
    }

    template <typename Clock, typename Duration>
    channel_op_status push_wait_until(value_type const& value,
                                      std::chrono::time_point<Clock, Duration> const& timeout_time_)
    {
        context* active_ctx                                = context::active();
        std::chrono::steady_clock::time_point timeout_time = detail::convert(timeout_time_);
        for (;;)
        {
            detail::spinlock_lock lk{splk_};
            if (BOOST_UNLIKELY(is_closed_()))
            {
                return channel_op_status::closed;
            }
            if (is_full_())
            {
                if (!waiting_producers_.suspend_and_wait_until(lk, active_ctx, timeout_time))
                {
                    return channel_op_status::timeout;
                }
            }
            else
            {
                slots_[pidx_] = value;
                pidx_         = (pidx_ + 1) % capacity_;
                waiting_consumers_.notify_one();
                return channel_op_status::success;
            }
        }
    }

    template <typename Clock, typename Duration>
    channel_op_status push_wait_until(value_type&& value, std::chrono::time_point<Clock, Duration> const& timeout_time_)
    {
        context* active_ctx                                = context::active();
        std::chrono::steady_clock::time_point timeout_time = detail::convert(timeout_time_);
        for (;;)
        {
            detail::spinlock_lock lk{splk_};
            if (BOOST_UNLIKELY(is_closed_()))
            {
                return channel_op_status::closed;
            }
            if (is_full_())
            {
                if (!waiting_producers_.suspend_and_wait_until(lk, active_ctx, timeout_time))
                {
                    return channel_op_status::timeout;
                }
            }
            else
            {
                slots_[pidx_] = std::move(value);
                pidx_         = (pidx_ + 1) % capacity_;
                // notify one waiting consumer
                waiting_consumers_.notify_one();
                return channel_op_status::success;
            }
        }
    }

    /**
     * Consumes an element from the ring buffer.  This operation will suspend until an element in
     * the ring buffer becomes available.
     */
    [[nodiscard]] auto read() -> ReadOperation
    {
        return ReadOperation{*this};
    }

    mrc::coroutines::Task<channel_op_status> read(value_type& value)
    {
        // context* active_ctx = context::active();
        for (;;)
        {
            detail::spinlock_lock lk{splk_};
            if (is_empty_())
            {
                if (BOOST_UNLIKELY(is_closed_()))
                {
                    co_return channel_op_status::closed;
                }
                // waiting_consumers_.suspend_and_wait(lk, active_ctx);
                co_await ReadOperation{*this};
            }
            else
            {
                value = std::move(slots_[cidx_]);
                cidx_ = (cidx_ + 1) % capacity_;
                waiting_producers_.notify_one();
                co_return channel_op_status::success;
            }
        }
    }

    // channel_op_status try_pop(value_type& value)
    // {
    //     detail::spinlock_lock lk{splk_};
    //     if (is_empty_())
    //     {
    //         return is_closed_() ? channel_op_status::closed : channel_op_status::empty;
    //     }
    //     value = std::move(slots_[cidx_]);
    //     cidx_ = (cidx_ + 1) % capacity_;
    //     waiting_producers_.notify_one();
    //     return channel_op_status::success;
    // }

    // channel_op_status pop(value_type& value)
    // {
    //     context* active_ctx = context::active();
    //     for (;;)
    //     {
    //         detail::spinlock_lock lk{splk_};
    //         if (is_empty_())
    //         {
    //             if (BOOST_UNLIKELY(is_closed_()))
    //             {
    //                 return channel_op_status::closed;
    //             }
    //             waiting_consumers_.suspend_and_wait(lk, active_ctx);
    //         }
    //         else
    //         {
    //             value = std::move(slots_[cidx_]);
    //             cidx_ = (cidx_ + 1) % capacity_;
    //             waiting_producers_.notify_one();
    //             return channel_op_status::success;
    //         }
    //     }
    // }

    // value_type value_pop()
    // {
    //     context* active_ctx = context::active();
    //     for (;;)
    //     {
    //         detail::spinlock_lock lk{splk_};
    //         if (is_empty_())
    //         {
    //             if (BOOST_UNLIKELY(is_closed_()))
    //             {
    //                 throw fiber_error{std::make_error_code(std::errc::operation_not_permitted),
    //                                   "boost fiber: channel is closed"};
    //             }
    //             waiting_consumers_.suspend_and_wait(lk, active_ctx);
    //         }
    //         else
    //         {
    //             value_type value = std::move(slots_[cidx_]);
    //             cidx_            = (cidx_ + 1) % capacity_;
    //             waiting_producers_.notify_one();
    //             return value;
    //         }
    //     }
    // }

    // template <typename Rep, typename Period>
    // channel_op_status pop_wait_for(value_type& value, std::chrono::duration<Rep, Period> const& timeout_duration)
    // {
    //     return pop_wait_until(value, std::chrono::steady_clock::now() + timeout_duration);
    // }

    // template <typename Clock, typename Duration>
    // channel_op_status pop_wait_until(value_type& value, std::chrono::time_point<Clock, Duration> const&
    // timeout_time_)
    // {
    //     context* active_ctx                                = context::active();
    //     std::chrono::steady_clock::time_point timeout_time = detail::convert(timeout_time_);
    //     for (;;)
    //     {
    //         detail::spinlock_lock lk{splk_};
    //         if (is_empty_())
    //         {
    //             if (BOOST_UNLIKELY(is_closed_()))
    //             {
    //                 return channel_op_status::closed;
    //             }
    //             if (!waiting_consumers_.suspend_and_wait_until(lk, active_ctx, timeout_time))
    //             {
    //                 return channel_op_status::timeout;
    //             }
    //         }
    //         else
    //         {
    //             value = std::move(slots_[cidx_]);
    //             cidx_ = (cidx_ + 1) % capacity_;
    //             waiting_producers_.notify_one();
    //             return channel_op_status::success;
    //         }
    //     }
    // }

    class iterator
    {
      private:
        typedef typename std::aligned_storage<sizeof(value_type), alignof(value_type)>::type storage_type;

        buffered_channel_fibers_to_coro* chan_{nullptr};
        storage_type storage_;

        void increment_(bool initial = false)
        {
            BOOST_ASSERT(nullptr != chan_);
            try
            {
                if (!initial)
                {
                    reinterpret_cast<value_type*>(std::addressof(storage_))->~value_type();
                }
                ::new (static_cast<void*>(std::addressof(storage_))) value_type{chan_->value_pop()};
            } catch (fiber_error const&)
            {
                chan_ = nullptr;
            }
        }

      public:
        using iterator_category = std::input_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using pointer           = value_type*;
        using reference         = value_type&;

        using pointer_t   = pointer;
        using reference_t = reference;

        iterator() = default;

        explicit iterator(buffered_channel_fibers_to_coro<T>* chan) noexcept : chan_{chan}
        {
            increment_(true);
        }

        iterator(iterator const& other) noexcept : chan_{other.chan_} {}

        iterator& operator=(iterator const& other) noexcept
        {
            if (BOOST_LIKELY(this != &other))
            {
                chan_ = other.chan_;
            }
            return *this;
        }

        bool operator==(iterator const& other) const noexcept
        {
            return other.chan_ == chan_;
        }

        bool operator!=(iterator const& other) const noexcept
        {
            return other.chan_ != chan_;
        }

        iterator& operator++()
        {
            reinterpret_cast<value_type*>(std::addressof(storage_))->~value_type();
            increment_();
            return *this;
        }

        const iterator operator++(int) = delete;

        reference_t operator*() noexcept
        {
            return *reinterpret_cast<value_type*>(std::addressof(storage_));
        }

        pointer_t operator->() noexcept
        {
            return reinterpret_cast<value_type*>(std::addressof(storage_));
        }
    };

    friend class iterator;
};

template <typename T>
typename buffered_channel_fibers_to_coro<T>::iterator begin(buffered_channel_fibers_to_coro<T>& chan)
{
    return typename buffered_channel_fibers_to_coro<T>::iterator(&chan);
}

template <typename T>
typename buffered_channel_fibers_to_coro<T>::iterator end(buffered_channel_fibers_to_coro<T>&)
{
    return typename buffered_channel_fibers_to_coro<T>::iterator();
}

}  // namespace fibers
}  // namespace boost

#ifdef BOOST_HAS_ABI_HEADERS
    #include BOOST_ABI_SUFFIX
#endif

#endif  // BOOST_FIBERS_BUFFERED_CHANNEL_H
