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
#include "buffered_channel_fibers_to_coro.hpp"
#include "mrc/node/sink_properties.hpp"
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
#include <mrc/node/sink_channel_owner.hpp>
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
            m_future = boost::fibers::async(boost::fibers::launch::post, std::move(fn), std::forward<ArgsT>(args)...);

            // m_inner_fn = [fn       = std::move(fn),
            //               ... args = std::forward<ArgsT>(args)](std::coroutine_handle<> continuation) mutable {
            //     // Call the function
            //     auto result = fn(std::forward<ArgsT>(args)...);

            //     // Set the result into the promise
            //     // promise.set_value(std::move(result));

            //     // Resume the coroutine
            //     continuation.resume();
            // };
        }

        bool await_ready() noexcept
        {
            return false;
        }

        bool await_suspend(std::coroutine_handle<> continuation) noexcept
        {
            // Launch the fiber
            // m_future = boost::fibers::async(boost::fibers::launch::post, m_inner_fn, std::move(continuation));

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

            // If its ready, return false to indicate that the coroutine has completed
            // return !(m_future.wait_for(std::chrono::milliseconds::zero()) == boost::fibers::future_status::ready);
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

template <typename InputT, typename OutputT>
class CoroutineRunnableSubscriber
{
  public:
    using reader_awaiter_t = BoostFutureAwaiter<mrc::channel::Status(InputT&)>;
    using writer_awaiter_t = BoostFutureAwaiter<mrc::channel::Status(OutputT&&)>;

    CoroutineRunnableSubscriber() = default;

    ~CoroutineRunnableSubscriber()
    {
        VLOG(10) << "In CoroutineRunnableSubscriber constructor";
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
    CoroutineRunnable()           = default;
    ~CoroutineRunnable() override = default;

  private:
    void run(mrc::runnable::Context& ctx) override;
    void on_state_update(const state_t& state) final;

    Task<void> main_task();

    virtual Task<void> on_data(InputT&& value, CoroutineRunnableSubscriber<InputT, OutputT>& subscriber) = 0;

    virtual Task<void> do_work(CoroutineRunnableSubscriber<InputT, OutputT>& subscriber) = 0;

    bool m_is_running{false};

    mrc::pymrc::PyHolder m_loop;

    std::unique_ptr<CoroutineRunnableSubscriber<InputT, OutputT>> m_subscriber;
};

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    py::gil_scoped_acquire gil;

    //
    py::print("Creating loop");

    // Gets (or more likely, creates) an event loop and runs it forever until stop is called
    m_loop = py::module_::import("asyncio").attr("new_event_loop")();

    py::print("Setting loop current");

    // Set the event loop as the current event loop
    py::module::import("asyncio").attr("set_event_loop")(m_loop);

    py::print("Running forever");

    // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
    m_loop.attr("run_until_complete")(mrc::pycoro::BoostFibersMainPyAwaitable(this->main_task()));

    py::print("Done running forever");
}

template <typename InputT, typename OutputT>
Task<void> CoroutineRunnable<InputT, OutputT>::main_task()
{
    auto read_awaiter = BoostFutureAwaiter(std::function{[this](InputT& value) {
        return this->get_readable_edge()->await_read(value);
    }});

    auto write_awaiter = BoostFutureAwaiter(std::function{[this](OutputT&& value) {
        return this->get_writable_edge()->await_write(std::move(value));
    }});

    m_subscriber = std::make_unique<CoroutineRunnableSubscriber<InputT, OutputT>>(std::move(read_awaiter),
                                                                                  std::move(write_awaiter));

    // auto main_generator = [this]() -> mrc::coroutines::AsyncGenerator<InputT> {
    //     while (m_is_running)
    //     {
    //         InputT value;

    //         // Pull a message off of the upstream channel
    //         auto status = co_await read_awaiter(std::ref(value));

    //         if (status != mrc::channel::Status::success)
    //         {
    //             break;
    //         }

    //         co_yield std::move(value);
    //     }

    //     co_return;
    // };

    auto& subscriber = *m_subscriber;

    auto iter = co_await subscriber.begin();

    while (iter != subscriber.end())
    {
        auto data = std::move(*iter);

        // {
        //     py::gil_scoped_acquire gil;

        //     // Push the value into the coroutine
        //     py::module_::import("asyncio").attr("ensure_future")(
        //         mrc::pycoro::CppToPyAwaitable(this->on_data(std::move(data), *m_subscriber)));
        // }

        co_await this->on_data(std::move(data), *m_subscriber);

        // Advance the iterator
        co_await ++iter;
    }

    // co_await this->do_work(*m_subscriber);

    VLOG(10) << "CoroutineRunnable dropping edge connections";

    // Need to drop the output edges
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
void CoroutineRunnable<InputT, OutputT>::on_state_update(const state_t& state)
{
    switch (state)
    {
    case state_t::Stop:
        DCHECK(m_subscriber) << "Should never be null once started";
        // Do nothing, we wait for the upstream channel to return closed
        m_subscriber->stop();
        break;

    case state_t::Kill:
        DCHECK(m_subscriber) << "Should never be null once started";

        m_subscriber->kill();
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
    Task<void> on_data(std::shared_ptr<ControlMessage>&& data,
                       CoroutineRunnableSubscriber<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>&
                           subscriber) override
    {
        VLOG(10) << "Got message in PyLLMEngineStage. Calling LLMEnging::run()";

        auto result = co_await m_engine->run(std::move(data));

        // Push the output messages
        for (auto& out_message : result)
        {
            co_await subscriber.await_write(std::move(out_message));
        }
    }

    Task<void> do_work(CoroutineRunnableSubscriber<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>&
                           subscriber) override
    {
        for (auto iter = co_await subscriber.begin(); iter != subscriber.end(); co_await ++iter)
        {
            VLOG(10) << "Got message in PyLLMEngineStage. Calling LLMEnging::run()";

            auto result = co_await m_engine->run(*iter);

            // Push the output messages
            for (auto& out_message : result)
            {
                co_await subscriber.await_write(std::move(out_message));
            }
        }

        VLOG(10) << "PyLLMEngineStage::do_work() finished";
    }

    std::shared_ptr<LLMEngine> m_engine;
};

}  // namespace morpheus::llm
