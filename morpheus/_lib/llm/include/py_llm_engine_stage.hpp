/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once
#include "async_generator.hpp"
#include "closable_ring_buffer.hpp"
#include "py_llm_node.hpp"
#include "pycoro/pycoro.hpp"
#include "schedule_on.hpp"

#include "morpheus/export.h"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_task_handler.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/types.hpp"

#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <boost/fiber/future/promise.hpp>
#include <boost/fiber/policy.hpp>
#include <glog/logging.h>
#include <mrc/channel/status.hpp>
#include <mrc/coroutines/concepts/awaitable.hpp>
#include <mrc/coroutines/detail/void_value.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/node/sink_channel_owner.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/node/source_channel_owner.hpp>
#include <mrc/node/source_properties.hpp>
#include <mrc/runnable/context.hpp>
#include <mrc/runnable/forward.hpp>
#include <mrc/runnable/runnable.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <stop_token>
#include <utility>

namespace mrc::coroutines {
/**
 * @brief Scheduler base class
 *
 * Allows all schedulers to be discovered via the mrc::this_thread::current_scheduler()
 */
class Scheduler
{
  public:
    struct Operation
    {
        Operation(Scheduler& scheduler) : m_scheduler(scheduler) {}

        constexpr static auto await_ready() noexcept -> bool
        {
            return false;
        }

        auto await_suspend(std::coroutine_handle<> awaiting_coroutine)
        {
            m_awaiting_coroutine = awaiting_coroutine;
            m_scheduler.schedule_operation(this);
        }

        void await_resume()
        {
            m_scheduler.resume_operation(this);
        }

        Scheduler& m_scheduler;
        std::coroutine_handle<> m_awaiting_coroutine;
        Operation* m_next{nullptr};
    };

    Scheduler()          = default;
    virtual ~Scheduler() = default;

    // /**
    //  * Schedules the currently executing coroutine to be run on this thread pool.  This must be
    //  * called from within the coroutines function body to schedule the coroutine on the thread pool.
    //  * @throw std::runtime_error If the thread pool is `shutdown()` scheduling new tasks is not permitted.
    //  * @return The operation to switch from the calling scheduling thread to the executor thread
    //  *         pool thread.
    //  */
    [[nodiscard]] virtual Operation schedule() = 0;

    // Runs the task until its complete
    virtual void run_until_complete(Task<> task) = 0;

  private:
    virtual std::coroutine_handle<> schedule_operation(Operation* operation) = 0;
    virtual void resume_operation(Operation* operation)                      = 0;
};
}  // namespace mrc::coroutines

namespace morpheus::llm {
namespace py = pybind11;

class PythonAsyncioScheduler : public mrc::coroutines::Scheduler
{
  public:
    PythonAsyncioScheduler(size_t concurrency) : m_concurrency(concurrency) {}

    [[nodiscard]] Operation schedule() override
    {
        return Operation{*this};
    }

    mrc::pymrc::PyHolder init_loop()
    {
        std::unique_lock lock(m_mutex);

        if (m_loop)
        {
            return m_loop;
        }

        py::gil_scoped_acquire gil;

        // Otherwise check if one is already allocated
        auto loop = py::module_::import("asyncio").attr("get_running_loop")();

        if (!loop)
        {
            // Need to create a loop
            LOG(INFO) << "CoroutineRunnable::run() > Creating new event loop";

            // Gets (or more likely, creates) an event loop and runs it forever until stop is called
            loop = py::module_::import("asyncio").attr("new_event_loop")();

            // Set the event loop as the current event loop
            py::module::import("asyncio").attr("set_event_loop")(loop);
        }

        m_loop = std::move(loop);

        return m_loop;
    }

    // Runs the task until its complete
    void run_until_complete(Task<> task) override
    {
        auto loop = this->init_loop();

        LOG(INFO) << "CoroutineRunnable::run() > Calling run_until_complete() on main_task()";

        // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
        loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(std::move(task)));

        LOG(INFO)
            << "CoroutineRunnable::run() > run_until_complete() returned. Waiting for all enqueued tasks to complete";

        std::unique_lock lock(m_mutex);

        // Block until all outstanding coroutines have completed
        m_cv.wait(lock, [this]() {
            return m_outstanding == 0;
        });
    }

  private:
    class Operation
    {
      public:
        Operation(PythonAsyncioScheduler& parent) noexcept : m_parent(parent) {}

        bool await_ready() noexcept
        {
            return false;
        }

        void await_suspend(std::coroutine_handle<> awaiting_coroutine)
        {
            std::unique_lock lock(m_parent.m_mutex);

            m_awaiting_coroutine = awaiting_coroutine;

            // Check if we have less than the number of concurrent coroutines
            if (m_parent.m_outstanding < m_parent.m_concurrency)
            {
                this->enqueue_on_loop(lock);
            }
            else
            {
                // Add to the queue of waiting operations
                m_parent.m_waiting_operations.push_back(this);
            }
        }

        void await_resume() noexcept
        {
            std::unique_lock lock(m_parent.m_mutex);

            // Decrement the number of outstanding coroutines
            this->m_parent.m_outstanding--;

            m_parent.m_cv.notify_all();

            // Check if we have any more operations to schedule
            if (!m_parent.m_waiting_operations.empty())
            {
                // Remove the operation from the queue
                auto* to_resume = m_parent.m_waiting_operations.front();
                m_parent.m_waiting_operations.pop_front();

                // Drop the lock before resuming
                lock.unlock();

                // Resume the next operation
                to_resume->enqueue_on_loop(lock);
            }
        }

      private:
        // lock is not used but helps make it clear that it should be held
        void enqueue_on_loop(std::unique_lock<std::mutex>& lock)
        {
            py::gil_scoped_acquire gil;

            auto asyncio_mod = py::module_::import("asyncio");

            // Increment the number of outstanding coroutines
            ++m_parent.m_outstanding;

            auto& loop = m_parent.get_loop();

            // TODO(MDD): Check whether or not we need thread safe version
            loop.attr("call_soon_threadsafe")(
                py::cpp_function([this, awaiting_coroutine = std::move(m_awaiting_coroutine)]() {
                    awaiting_coroutine.resume();
                }));
        }

        friend class PythonAsyncioScheduler;

        PythonAsyncioScheduler& m_parent;
        std::coroutine_handle<> m_awaiting_coroutine;
        Operation* m_next{nullptr};
    };

    mrc::pymrc::PyHolder& get_loop()
    {
        // TODO(MDD): Check that we are on the same thread as the loop
        return m_loop;
    }

    std::mutex m_mutex;
    std::condition_variable m_cv;

    size_t m_concurrency{8};

    std::atomic_size_t m_outstanding{0};
    Operation* m_waiting_operation{nullptr};
    Operation* m_waiting_operation_back{nullptr};
    std::list<Operation*> m_waiting_operations;

    mrc::pymrc::PyHolder m_loop;
};

template <typename SignatureT>
class BoostFutureAwaiter
{
    class Awaiter;

  public:
    BoostFutureAwaiter(std::function<SignatureT> fn) : m_fn(std::move(fn)) {}

    // template <typename LambdaT>
    // BoostFutureAwaiter(LambdaT&& fn) : m_fn(std::function{std::forward<LambdaT>(fn)})
    // {}

    template <typename... ArgsT>
    auto operator()(ArgsT&&... args) -> Awaiter
    {
        // Make a copy of m_fn here so we can call this operator again
        return Awaiter(m_fn, std::forward<ArgsT>(args)...);
    }

  private:
    class Awaiter
    {
      public:
        using return_t = std::function<SignatureT>::result_type;

        template <typename... ArgsT>
        Awaiter(std::function<SignatureT> fn, ArgsT&&... args)
        {
            m_future = boost::fibers::async(boost::fibers::launch::post, fn, std::forward<ArgsT>(args)...);
        }

        bool await_ready() noexcept
        {
            return false;
        }

        bool await_suspend(std::coroutine_handle<> continuation) noexcept
        {
            // Launch a new fiber that waits on the future and then resumes the coroutine
            boost::fibers::async(
                boost::fibers::launch::post,
                [this](std::coroutine_handle<> continuation) {
                    // Wait on the future
                    m_future.wait();

                    // Resume the coroutine
                    continuation.resume();
                },
                std::move(continuation));

            return true;
        }

        auto await_resume() noexcept
        {
            return m_future.get();
        }

      private:
        boost::fibers::future<return_t> m_future;
        std::function<void(std::coroutine_handle<>)> m_inner_fn;
    };

    std::function<SignatureT> m_fn;
};

template <typename T>
class IReadable
{
  public:
    virtual ~IReadable()                                    = default;
    virtual Task<mrc::channel::Status> async_read(T& value) = 0;
};

template <typename T>
class BoostFutureReader : public IReadable<T>
{
  public:
    template <typename FuncT>
    BoostFutureReader(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    Task<mrc::channel::Status> async_read(T& value) override
    {
        co_return co_await m_awaiter(std::ref(value));
    }

  private:
    BoostFutureAwaiter<mrc::channel::Status(T&)> m_awaiter;
};

template <typename T>
class IWritable
{
  public:
    virtual ~IWritable()                                      = default;
    virtual Task<mrc::channel::Status> async_write(T&& value) = 0;
};

template <typename T>
class BoostFutureWriter : public IWritable<T>
{
  public:
    template <typename FuncT>
    BoostFutureWriter(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    Task<mrc::channel::Status> async_write(T&& value) override
    {
        co_return co_await m_awaiter(std::move(value));
    }

  private:
    BoostFutureAwaiter<mrc::channel::Status(T&&)> m_awaiter;
};

template <typename InputT, typename OutputT>
class CoroutineRunnableSubscriber
{
  public:
    using reader_awaiter_t = BoostFutureAwaiter<mrc::channel::Status(InputT&)>;
    using writer_awaiter_t = BoostFutureAwaiter<mrc::channel::Status(OutputT&&)>;

    CoroutineRunnableSubscriber() = default;

    ~CoroutineRunnableSubscriber()
    {
        VLOG(10) << "In CoroutineRunnableSubscriber destructor";
    }

    CoroutineRunnableSubscriber(reader_awaiter_t reader_awaiter, writer_awaiter_t writer_awaiter) :
      m_reader_awaiter(std::move(reader_awaiter)),
      m_writer_awaiter(std::move(writer_awaiter))
    {
        m_generator = [](reader_awaiter_t& read_awaiter,
                         std::stop_token stop_token) -> mrc::coroutines::AsyncGenerator<InputT> {
            while (!stop_token.stop_requested())
            {
                InputT value;

                // Pull a message off of the upstream channel
                auto status = co_await read_awaiter(std::ref(value));

                if (status != mrc::channel::Status::success)
                {
                    break;
                }

                co_yield std::move(value);
            }

            co_return;
        }(m_reader_awaiter, m_stop_source.get_token());
    }

    auto await_write(OutputT&& value)
    {
        return m_writer_awaiter(std::move(value));
    }

    void stop()
    {
        // m_is_running = false;
    }

    void kill()
    {
        // Do anything special?
    }

    auto begin() noexcept
    {
        return m_generator.begin();
    }

    auto end() noexcept
    {
        return m_generator.end();
    }

  private:
    std::stop_source m_stop_source;
    bool m_is_running{true};
    std::string m_name{"Let's see if this works"};

    mrc::coroutines::AsyncGenerator<InputT> m_generator;

    reader_awaiter_t m_reader_awaiter;
    writer_awaiter_t m_writer_awaiter;
};

template <typename T>
class CoroutineRunnableSink : public mrc::node::WritableProvider<T>,
                              public mrc::node::ReadableAcceptor<T>,
                              public mrc::node::SinkChannelOwner<T>
{
  protected:
    CoroutineRunnableSink()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    auto build_readable_generator(std::stop_token stop_token) -> mrc::coroutines::AsyncGenerator<T>
    {
        auto read_awaiter = BoostFutureReader<T>([this](T& value) {
            return this->get_readable_edge()->await_read(value);
        });

        while (!stop_token.stop_requested())
        {
            T value;

            // Pull a message off of the upstream channel
            auto status = co_await read_awaiter.async_read(std::ref(value));

            if (status != mrc::channel::Status::success)
            {
                break;
            }

            co_yield std::move(value);
        }

        co_return;
    }
};

template <typename T>
class CoroutineRunnableSource : public mrc::node::WritableAcceptor<T>,
                                public mrc::node::ReadableProvider<T>,
                                public mrc::node::SourceChannelOwner<T>
{
  protected:
    CoroutineRunnableSource()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    // auto build_readable_generator(std::stop_token stop_token)
    //     -> mrc::coroutines::AsyncGenerator<mrc::coroutines::detail::VoidValue>
    // {
    //     while (!stop_token.stop_requested())
    //     {
    //         co_yield mrc::coroutines::detail::VoidValue{};
    //     }

    //     co_return;
    // }

    auto build_writable_receiver() -> std::unique_ptr<IWritable<T>>
    {
        return std::make_unique<BoostFutureWriter<T>>([this](T&& value) {
            return this->get_writable_edge()->await_write(std::move(value));
        });
    }
};

template <typename InputT, typename OutputT>
class CoroutineRunnable : public CoroutineRunnableSink<InputT>,
                          public CoroutineRunnableSource<OutputT>,
                          public mrc::runnable::RunnableWithContext<>
{
    using state_t = mrc::runnable::Runnable::State;

  public:
    CoroutineRunnable(size_t concurrency = 128) : m_concurrency(concurrency){};
    ~CoroutineRunnable() override = default;

  private:
    void run(mrc::runnable::Context& ctx) override;
    void on_state_update(const state_t& state) final;

    Task<void> main_task(mrc::coroutines::Scheduler& scheduler);

    Task<void> process_one(InputT&& value, IWritable<OutputT>& writer);

    virtual mrc::coroutines::AsyncGenerator<OutputT> on_data(InputT&& value) = 0;

    std::stop_source m_stop_source;

    size_t m_concurrency{8};
};

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    // auto& scheduler = ctx.scheduler();
    auto scheduler = PythonAsyncioScheduler(m_concurrency);

    scheduler.run_until_complete(this->main_task(scheduler));

    VLOG(10) << "CoroutineRunnable dropping edge connections";

    // Need to drop the output edges
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();

    // py::gil_scoped_acquire gil;

    // LOG(INFO) << "CoroutineRunnable::run() > Creating new event loop";

    // // Gets (or more likely, creates) an event loop and runs it forever until stop is called
    // auto loop = py::module_::import("asyncio").attr("new_event_loop")();

    // // Set the event loop as the current event loop
    // py::module::import("asyncio").attr("set_event_loop")(loop);

    // LOG(INFO) << "CoroutineRunnable::run() > Calling run_until_complete() on main_task()";

    // // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
    // loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(this->main_task()));

    // LOG(INFO) << "CoroutineRunnable::run() > run_until_complete() returned. Exiting run()";
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::main_task(mrc::coroutines::Scheduler& scheduler)
{
    // Get the generator and receiver
    auto input_generator = CoroutineRunnableSink<InputT>::build_readable_generator(m_stop_source.get_token());
    auto output_receiver = CoroutineRunnableSource<OutputT>::build_writable_receiver();

    auto iter = co_await input_generator.begin();

    while (iter != input_generator.end())
    {
        // Push the value into the coroutine
        auto output_generator =
            mrc::coroutines::schedule_on(scheduler, this->process_one(std::move(*iter), *output_receiver));

        // Advance the iterator
        co_await ++iter;
    }

    // auto read_awaiter = BoostFutureReader<InputT>([this](InputT& value) {
    //     return this->get_readable_edge()->await_read(value);
    // });

    // auto write_awaiter = BoostFutureWriter<OutputT>([this](OutputT&& value) {
    //     return this->get_writable_edge()->await_write(std::move(value));
    // });

    // auto main_generator = [](std::stop_token stop_token,
    //                          IReadable<InputT>& reader) -> mrc::coroutines::AsyncGenerator<InputT> {
    //     while (!stop_token.stop_requested())
    //     {
    //         InputT value;

    //         // Pull a message off of the upstream channel
    //         auto status = co_await reader.async_read(std::ref(value));

    //         if (status != mrc::channel::Status::success)
    //         {
    //             break;
    //         }

    //         co_yield std::move(value);
    //     }

    //     co_return;
    // }(m_stop_source.get_token(), read_awaiter);

    // // Use a work queue to limit the number of concurrent coroutines
    // mrc::coroutines::ClosableRingBuffer<char> work_queue{{.capacity = m_concurrency}};

    // auto iter = co_await main_generator.begin();

    // while (iter != main_generator.end())
    // {
    //     // Get an element from the work queue
    //     co_await work_queue.write(0);

    //     if (m_concurrency > 1)
    //     {
    //         py::gil_scoped_acquire gil;

    //         // Push the value into the coroutine
    //         py::module_::import("asyncio").attr("ensure_future")(
    //             mrc::pycoro::CppToPyAwaitable(this->process_one(std::move(*iter), write_awaiter, work_queue)));
    //     }
    //     else
    //     {
    //         co_await this->process_one(std::move(*iter), write_awaiter, work_queue);
    //     }

    //     // Advance the iterator
    //     co_await ++iter;
    // }

    // // Close the queue to signal that we are done
    // work_queue.close();

    // co_await work_queue.completed();

    // VLOG(10) << "CoroutineRunnable dropping edge connections";

    // // Need to drop the output edges
    // mrc::node::SourceProperties<InputT>::release_edge_connection();
    // mrc::node::SinkProperties<OutputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::process_one(InputT&& value, IWritable<OutputT>& writer)
{
    // Call the on_data function
    auto on_data_gen = this->on_data(std::move(value));

    auto iter = co_await on_data_gen.begin();

    while (iter != on_data_gen.end())
    {
        co_await writer.async_write(std::move(*iter));

        // Advance the iterator
        co_await ++iter;
    }

    co_return;
}

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::on_state_update(const state_t& state)
{
    switch (state)
    {
    case state_t::Stop:
        // Do nothing, we wait for the upstream channel to return closed
        // m_stop_source.request_stop();
        break;

    case state_t::Kill:

        m_stop_source.request_stop();
        break;

    default:
        break;
    }
}

class MORPHEUS_EXPORT PyLLMEngineStage
  : public CoroutineRunnable<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    PyLLMEngineStage(std::shared_ptr<LLMEngine> engine) : m_engine(std::move(engine)) {}

    ~PyLLMEngineStage() override = default;

    static std::shared_ptr<mrc::segment::Object<PyLLMEngineStage>> init(mrc::segment::Builder& builder,
                                                                        const std::string& name,
                                                                        std::shared_ptr<LLMEngine> engine)
    {
        auto stage = builder.construct_object<PyLLMEngineStage>(name, std::move(engine));

        return stage;
    }

  private:
    mrc::coroutines::AsyncGenerator<std::shared_ptr<ControlMessage>> on_data(
        std::shared_ptr<ControlMessage>&& data) override
    {
        VLOG(10) << "Got message in PyLLMEngineStage. Calling LLMEnging::run()";

        auto result = co_await m_engine->run(std::move(data));

        // Push the output messages
        for (auto&& out_message : result)
        {
            co_yield std::move(out_message);
        }

        co_return;
    }

    std::shared_ptr<LLMEngine> m_engine;
};

}  // namespace morpheus::llm
