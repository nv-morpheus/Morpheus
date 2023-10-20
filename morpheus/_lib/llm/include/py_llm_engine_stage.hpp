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
#include "py_llm_node.hpp"
#include "pycoro/pycoro.hpp"

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
#include <mrc/coroutines/async_generator.hpp>
#include <mrc/coroutines/closable_ring_buffer.hpp>
#include <mrc/coroutines/concepts/awaitable.hpp>
#include <mrc/coroutines/detail/void_value.hpp>
#include <mrc/coroutines/schedule_on.hpp>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
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
#include <pymrc/utilities/acquire_gil.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <stop_token>
#include <utility>

namespace morpheus::llm {
namespace py = pybind11;

class PythonAsyncioScheduler : public mrc::coroutines::Scheduler
{
  public:
    PythonAsyncioScheduler(size_t concurrency) {}

    std::string description() const override
    {
        return "PythonAsyncioScheduler";
    }

    void resume(std::coroutine_handle<> coroutine) override
    {
        if (coroutine.done())
        {
            LOG(WARNING) << "PythonAsyncioScheduler::resume() > Attempted to resume a completed coroutine";
            return;
        }

        py::gil_scoped_acquire gil;

        auto& loop = this->get_loop();

        // TODO(MDD): Check whether or not we need thread safe version
        loop.attr("call_soon_threadsafe")(py::cpp_function([this, handle = std::move(coroutine)]() {
            if (handle.done())
            {
                LOG(WARNING) << "PythonAsyncioScheduler::resume() > Attempted to resume a completed coroutine";
                return;
            }

            py::gil_scoped_release nogil;

            handle.resume();
        }));
    }

    mrc::pymrc::PyHolder& init_loop()
    {
        CHECK_EQ(PyGILState_Check(), 1) << "Must have the GIL when calling PythonAsyncioScheduler::init_loop()";

        std::unique_lock lock(m_mutex);

        if (m_loop)
        {
            return m_loop;
        }

        auto asyncio_mod = py::module_::import("asyncio");

        py::object loop;

        try
        {
            // Otherwise check if one is already allocated
            loop = asyncio_mod.attr("get_running_loop")();
        } catch (std::runtime_error&)
        {
            // Need to create a loop
            LOG(INFO) << "CoroutineRunnable::run() > Creating new event loop";

            // Gets (or more likely, creates) an event loop and runs it forever until stop is called
            loop = asyncio_mod.attr("new_event_loop")();

            // Set the event loop as the current event loop
            asyncio_mod.attr("set_event_loop")(loop);
        }

        m_loop = std::move(loop);

        return m_loop;
    }

    // Runs the task until its complete
    void run_until_complete(Task<void>&& task)
    {
        mrc::pymrc::AcquireGIL gil;

        auto& loop = this->init_loop();

        LOG(INFO) << "CoroutineRunnable::run() > Calling run_until_complete() on main_task()";

        // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
        loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(std::move(task)));

        LOG(INFO)
            << "CoroutineRunnable::run() > run_until_complete() returned. Waiting for all enqueued tasks to complete ";

        // Now wait until all tasks have been processed
        loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(
            this->get_task_container().garbage_collect_and_yield_until_empty()));
    }

  private:
    std::coroutine_handle<> schedule_operation(Operation* operation) override
    {
        this->resume(std::move(operation->m_awaiting_coroutine));

        return std::noop_coroutine();
    }

    mrc::pymrc::PyHolder& get_loop()
    {
        if (!m_loop)
        {
            throw std::runtime_error("Must call init_loop() before get_loop()");
        }

        // TODO(MDD): Check that we are on the same thread as the loop
        return m_loop;
    }

    std::mutex m_mutex;

    std::atomic_size_t m_outstanding{0};

    mrc::pymrc::PyHolder m_loop;
};

template <typename SignatureT>
class BoostFutureAwaiter
{
    class Awaiter;

  public:
    BoostFutureAwaiter(std::function<SignatureT> fn) : m_fn(std::move(fn)) {}

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

    auto build_writable_receiver() -> std::shared_ptr<IWritable<T>>
    {
        return std::make_shared<BoostFutureWriter<T>>([this](T&& value) {
            return this->get_writable_edge()->await_write(std::move(value));
        });
    }
};

template <typename InputT, typename OutputT>
class CoroutineRunnable : public CoroutineRunnableSink<InputT>,
                          public CoroutineRunnableSource<OutputT>,
                          public mrc::runnable::RunnableWithContext<>
{
    using state_t       = mrc::runnable::Runnable::State;
    using task_buffer_t = mrc::coroutines::ClosableRingBuffer<size_t>;

  public:
    CoroutineRunnable(size_t concurrency = 8) : m_concurrency(concurrency){};
    ~CoroutineRunnable() override = default;

  private:
    void run(mrc::runnable::Context& ctx) override;
    void on_state_update(const state_t& state) final;

    Task<void> main_task(mrc::coroutines::Scheduler& scheduler);

    Task<void> process_one(InputT&& value, std::shared_ptr<IWritable<OutputT>> writer, task_buffer_t& task_buffer);

    virtual mrc::coroutines::AsyncGenerator<OutputT> on_data(InputT&& value) = 0;

    std::stop_source m_stop_source;

    size_t m_concurrency{8};
};

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    // auto& scheduler = ctx.scheduler();

    // TODO(MDD): Eventually we should get this from the context object. For now, just create it directly
    auto scheduler = std::make_shared<PythonAsyncioScheduler>(m_concurrency);

    // Now use the scheduler to run the main task until it is complete
    scheduler->run_until_complete(this->main_task(*scheduler));

    // Need to drop the output edges
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::main_task(mrc::coroutines::Scheduler& scheduler)
{
    // Get the generator and receiver
    auto input_generator = CoroutineRunnableSink<InputT>::build_readable_generator(m_stop_source.get_token());
    auto output_receiver = CoroutineRunnableSource<OutputT>::build_writable_receiver();

    // Create the task buffer to limit the number of running tasks
    task_buffer_t task_buffer{{.capacity = m_concurrency}};

    size_t i = 0;

    auto iter = co_await input_generator.begin();

    while (iter != input_generator.end())
    {
        // Weird bug, cant directly move the value into the process_one call
        auto data = std::move(*iter);

        // Wait for an available slot in the task buffer
        co_await task_buffer.write(i);

        // Push the value into the coroutine. This may or may not block depending on how many outstanding tasks there
        // are
        scheduler.schedule(this->process_one(std::move(data), output_receiver, task_buffer));

        // Advance the iterator
        co_await ++iter;
        ++i;
    }

    // Close the buffer
    task_buffer.close();

    // Now block until all tasks are complete
    co_await task_buffer.completed();
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::process_one(InputT&& value,
                                                           std::shared_ptr<IWritable<OutputT>> writer,
                                                           task_buffer_t& task_buffer)
{
    // Call the on_data function
    auto on_data_gen = this->on_data(std::move(value));

    auto iter = co_await on_data_gen.begin();

    while (iter != on_data_gen.end())
    {
        // Weird bug, cant directly move the value into the async_write call
        auto data = std::move(*iter);

        co_await writer->async_write(std::move(data));

        // Advance the iterator
        co_await ++iter;
    }

    // Return the slot to the task buffer
    co_await task_buffer.read();

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
