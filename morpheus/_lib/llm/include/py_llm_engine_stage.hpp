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
#include <pymrc/types.hpp>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <ratio>
#include <stop_token>
#include <utility>

namespace mrc::coroutines {
class Scheduler
{
  public:
    void run_until_complete(Task<>)
};
}  // namespace mrc::coroutines

namespace morpheus::llm {
namespace py = pybind11;

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

    Task<void> main_task(Scheduler& scheduler);

    Task<void> process_one(InputT&& value,
                           IWritable<OutputT>& writer,
                           mrc::coroutines::ClosableRingBuffer<char>& work_queue);

    virtual mrc::coroutines::AsyncGenerator<OutputT> on_data(InputT&& value) = 0;

    std::stop_source m_stop_source;

    size_t m_concurrency{8};
};

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    auto& scheduler = ctx.scheduler();

    scheduler.run_until_complete(this->main_task(scheduler));

    VLOG(10) << "CoroutineRunnable dropping edge connections";

    // Need to drop the output edges
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();

    py::gil_scoped_acquire gil;

    LOG(INFO) << "CoroutineRunnable::run() > Creating new event loop";

    // Gets (or more likely, creates) an event loop and runs it forever until stop is called
    auto loop = py::module_::import("asyncio").attr("new_event_loop")();

    // Set the event loop as the current event loop
    py::module::import("asyncio").attr("set_event_loop")(loop);

    LOG(INFO) << "CoroutineRunnable::run() > Calling run_until_complete() on main_task()";

    // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
    loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(this->main_task()));

    LOG(INFO) << "CoroutineRunnable::run() > run_until_complete() returned. Exiting run()";
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::main_task(Scheduler& scheduler)
{
    // Get the generator and receiver
    auto input_generator = []() -> mrc::coroutines::AsyncGenerator<InputT> {};
    auto output_receiver;

    auto iter = co_await input_generator.begin();

    while (iter != input_generator.end())
    {
        // Push the value into the coroutine
        auto output_generator = schedule_on(scheduler, this->process_one(std::move(*iter), output_receiver));

        // Advance the iterator
        co_await ++iter;
    }

    auto read_awaiter = BoostFutureReader<InputT>([this](InputT& value) {
        return this->get_readable_edge()->await_read(value);
    });

    auto write_awaiter = BoostFutureWriter<OutputT>([this](OutputT&& value) {
        return this->get_writable_edge()->await_write(std::move(value));
    });

    auto main_generator = [](std::stop_token stop_token,
                             IReadable<InputT>& reader) -> mrc::coroutines::AsyncGenerator<InputT> {
        while (!stop_token.stop_requested())
        {
            InputT value;

            // Pull a message off of the upstream channel
            auto status = co_await reader.async_read(std::ref(value));

            if (status != mrc::channel::Status::success)
            {
                break;
            }

            co_yield std::move(value);
        }

        co_return;
    }(m_stop_source.get_token(), read_awaiter);

    // Use a work queue to limit the number of concurrent coroutines
    mrc::coroutines::ClosableRingBuffer<char> work_queue{{.capacity = m_concurrency}};

    auto iter = co_await main_generator.begin();

    while (iter != main_generator.end())
    {
        // Get an element from the work queue
        co_await work_queue.write(0);

        if (m_concurrency > 1)
        {
            py::gil_scoped_acquire gil;

            // Push the value into the coroutine
            py::module_::import("asyncio").attr("ensure_future")(
                mrc::pycoro::CppToPyAwaitable(this->process_one(std::move(*iter), write_awaiter, work_queue)));
        }
        else
        {
            co_await this->process_one(std::move(*iter), write_awaiter, work_queue);
        }

        // Advance the iterator
        co_await ++iter;
    }

    // Close the queue to signal that we are done
    work_queue.close();

    co_await work_queue.completed();

    VLOG(10) << "CoroutineRunnable dropping edge connections";

    // Need to drop the output edges
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::process_one(InputT&& value,
                                                           IWritable<OutputT>& writer,
                                                           mrc::coroutines::ClosableRingBuffer<char>& work_queue)
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

    // and when completed, return the work item to allow more to be processed
    co_await work_queue.read();

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
