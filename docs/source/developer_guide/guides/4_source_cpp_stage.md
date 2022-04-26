<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 4. Creating a C++ Source Stage

For this example, we are going to add a C++ implementation for the `RabbitMQSourceStage` we designed in the Python examples. The Python implementation of this stage emits messages of the type `MessageMeta`; as such, our C++ implementation must do the same.

For communicating with [RabbitMQ](https://www.rabbitmq.com/) we will be using the [SimpleAmqpClient](https://github.com/alanxz/SimpleAmqpClient) library, and [libcudf](https://docs.rapids.ai/api/libcudf/stable/index.html) for constructing the `DataFrame`.

## Header Definition

Our includes section looks like:

```cpp
#include <morpheus/messages/meta.hpp>  // for MessageMeta
#include <neo/core/segment.hpp> // for Segment
#include <pyneo/node.hpp>  // for PythonSource

#include <cudf/io/types.hpp>  // for table_with_metadata
#include <SimpleAmqpClient/SimpleAmqpClient.h> // for AmqpClient::Channel::ptr_t

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>
```

Our namespace and class definition looks like this:

```cpp
namespace morpheus_rabbit {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace std::literals;
using namespace morpheus;

class RabbitMQSourceStage : public neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
    using base_t::source_type_t;
```

Our base class defines `source_type_t` as an alias for `Writer<std::shared_ptr<MessageMeta>>::writer_type_t` which we are going to use as it will appear in some of our function signatures. The way to think about `source_type_t` is that the stage we are writing emits objects of type `MessageMeta`.

Our constructor looks similar to the constructor of our Python class with the majority of the parameters being specific to communicating with RabbitMQ. In this case the default destructor is sufficient.

```cpp
RabbitMQSourceStage(const neo::Segment &segment,
                    const std::string &name,
                    const std::string &host,
                    const std::string &exchange,
                    const std::string &exchange_type = "fanout"s,
                    const std::string &queue_name    = ""s,
                    std::chrono::milliseconds poll_interval = 100ms);
~RabbitMQSourceStage() override = default;
```

Our class will require a few private methods

```cpp
neo::Observable<source_type_t> build_observable();
void source_generator(neo::Subscriber<source_type_t> &sub);
cudf::io::table_with_metadata from_json(const std::string &body) const;
void close();
```

The `build_observable` method is responsible for constructing a Neo `Observable` for our source type, the result of which will be passed into our base's constructor. A Neo `Observable` is constructed by passing it a reference to a function (typically a lambda) which receives a reference to a Neo `Subscriber`. Typically, this function is the center of a source stage, making calls to the `Subscriber`'s `on_next`, `on_error` and `on_completed` methods. For this example, the RabbitMQ specific logic was broken out into the `source_generator` method, which should be analogous to the `source_generator` method from the Python class, and will emit new messages into the pipeline by calling `subscriber.on_next(message)`.

The `from_json` method parses a JSON string to a cuDF [table_with_metadata](https://docs.rapids.ai/api/libcudf/stable/structcudf_1_1io_1_1table__with__metadata.html). Lastly, the `close` method disconnects from the RabbitMQ exchange.

We will also need three private attributes specific to our interactions with RabbitMQ, our polling interval, the name of the queue we are listening to, and a pointer to our channel object.

```cpp
std::chrono::milliseconds m_poll_interval;
std::string m_queue_name;
AmqpClient::Channel::ptr_t m_channel;
```

Wrapping it all together our header file should look like this:
`examples/rabbitmq/_lib/rabbitmq_source.hpp`

```cpp
#pragma once

#include <morpheus/messages/meta.hpp>  // for MessageMeta
#include <pyneo/node.hpp>  // for neo::pyneo::PythonSource

#include <cudf/io/types.hpp>  // for cudf::io::table_with_metadata

#include <SimpleAmqpClient/SimpleAmqpClient.h>  // for AmqpClient::Channel::ptr_t

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>

namespace morpheus_rabbit {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace std::literals;
using namespace morpheus;

class RabbitMQSourceStage : public neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
    using base_t::source_type_t;

    RabbitMQSourceStage(const neo::Segment &segment,
                        const std::string &name,
                        const std::string &host,
                        const std::string &exchange,
                        const std::string &exchange_type        = "fanout"s,
                        const std::string &queue_name           = ""s,
                        std::chrono::milliseconds poll_interval = 100ms);

    ~RabbitMQSourceStage() override = default;

  private:
    neo::Observable<source_type_t> build_observable();
    void source_generator(neo::Subscriber<source_type_t> &sub);
    cudf::io::table_with_metadata from_json(const std::string &body) const;
    void close();

    std::chrono::milliseconds m_poll_interval;
    std::string m_queue_name;
    AmqpClient::Channel::ptr_t m_channel;
};

/****** RabbitMQSourceStageInferenceProxy**********************/
/**
 * @brief Interface proxy, used to insulate Python bindings.
 */
struct RabbitMQSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a RabbitMQSourceStage, and return the result.
     */
    static std::shared_ptr<RabbitMQSourceStage> init(
        neo::Segment &segment,
        const std::string &name,
        const std::string &host,
        const std::string &exchange,
        const std::string &exchange_type,
        const std::string &queue_name,
        std::chrono::milliseconds poll_interval);
};
#pragma GCC visibility pop
}  // namespace morpheus_rabbit
```

## Source Definition

Our includes section looks like:

```cpp
#include "rabbitmq_source.hpp"

#include <neo/core/segment.hpp>
#include <neo/core/segment_object.hpp>

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>

#include <glog/logging.h>
#include <pybind11/chrono.h>  // for timedelta->chrono
#include <pybind11/pybind11.h>
#include <boost/fiber/operations.hpp>  // for this_fiber::sleep_for

#include <exception>
#include <sstream>
#include <vector>
```

The two Neo includes bringing in the actual definitions for Neo `Segment` and `SegmentObject`. The [Google Logging Library](https://github.com/google/glog) (glog) is used by Morpheus for logging; however, the choice of a logger is up to the individual developer.

Neo uses the [Boost.Fiber](https://www.boost.org/doc/libs/1_77_0/libs/fiber/doc/html/fiber/overview.html) library to perform task scheduling. In the future, Neo will likely expose a configuration option to choose between fibers or `std::thread`.
For now, all Morpheus stages, both Python and C++, are executed within a fiber. In general, authors of a stage don't need to be too concerned about this detail, with two notable exceptions:
1. Rather than yielding or sleeping a thread,  stage authors should instead call [boost::this_fiber::yield](https://www.boost.org/doc/libs/master/libs/fiber/doc/html/fiber/fiber_mgmt/this_fiber.html#this_fiber_yield) and [boost::this_fiber::sleep_for](https://www.boost.org/doc/libs/master/libs/fiber/doc/html/fiber/fiber_mgmt/this_fiber.html#this_fiber_sleep_for) respectively.
1. In cases where thread-local-storage is desired, [fiber local storage](https://www.boost.org/doc/libs/1_77_0/libs/fiber/doc/html/fiber/fls.html) should be used instead.

Authors of stages that require concurrency are free to choose their own concurrency models. Launching new processes, threads, or fibers from within a stage is permissible as long as the management of those resources is contained within the stage. Newly launched threads are also free to use thread-local storage so long as it doesn't occur within the thread the stage is executed from.

The definition for our constructor looks like this:

```cpp
RabbitMQSourceStage::RabbitMQSourceStage(const neo::Segment &segment,
                                         const std::string &name,
                                         const std::string &host,
                                         const std::string &exchange,
                                         const std::string &exchange_type,
                                         const std::string &queue_name,
                                         std::chrono::milliseconds poll_interval) :
  neo::SegmentObject(segment, name),
  base_t(segment, name, build_observable()),
  m_channel{AmqpClient::Channel::Create(host)},
  m_poll_interval{poll_interval}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}
```

The key thing to note is the third argument in the invocation of our base's constructor is our observable:

```cpp
base_t(segment, name, build_observable())
```

The observable argument to the constructor contains an empty default value, allowing stage authors to later define the observable by calling the `set_source_observable` method; the constructor could instead be written as:

```cpp
base_t(segment, name)
{
    this->set_source_observable(build_observable());
}
```

Our `build_observable` method returns an observable, which needs to do three things:
1. Emit data into the pipeline by calling `Subscriber`'s `on_next` method. In our example, this occurs in the `source_generator` method.
1. When an error occurs, call the `Subscriber`'s `on_error` method.
1. When we are done, call the `Subscriber`'s `on_complete` method.
Note: Some source stages, such as ones that read input data from a file, there is a clear point where the stage is complete. Others such as this one are intended to continue running until it is shut down. For these situations, the stage can poll the `Subscriber`'s `is_subscribed` method, which will return a value of `false` on shut-down.

```cpp
neo::Observable<RabbitMQSourceStage::source_type_t> RabbitMQSourceStage::build_observable()
{
    return neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t> &subscriber) {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception &e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    });
}
```

As a design decision, we left the majority of the RabbitMQ specific code in the `source_generator` method, leaving Morpheus specific code in the `build_observable` method. For each message that we receive and can successfully parse, we call the `Subscriber`'s `on_next` method. When there are no messages in the queue, we yield the fiber by sleeping, potentially allowing the scheduler to perform a context switch.

```cpp
void RabbitMQSourceStage::source_generator(neo::Subscriber<RabbitMQSourceStage::source_type_t> &subscriber)
{
    const std::string consumer_tag = m_connection->BasicConsume(m_queue_name, "", true, false);
    while (subscriber.is_subscribed())
    {
        AmqpClient::Envelope::ptr_t envelope;
        if (m_connection->BasicConsumeMessage(consumer_tag, envelope, 0))
        {
            try
            {
                auto table   = from_json(envelope->Message()->Body());
                auto message = MessageMeta::create_from_cpp(std::move(table), 0);
                subscriber.on_next(std::move(message));
            } catch (const std::exception &e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_connection->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            boost::this_fiber::sleep_for(m_poll_interval);
        }
    }
}
```

## A note on performance:

We don't yet know how large the messages we are going to be receiving from RabbitMQ, but we should assume that they may be quite large. As such, we try to limit the number of copies of this data, preferring to instead pass by reference or move data. The `SimpleAmqpClient`'s `Body()` method returns a const reference to the payload, which we also pass by reference into the `from_json` method. Since our stage has no need for the data itself after it's emitted into the pipeline, we move our cuDF data table when we construct our `MessageMeta` instance, and again, we then move the message into the subscriber's `on_next` method.

Our `from_json` and `close` methods are rather straight forward:

```cpp
cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string &body) const
{
    cudf::io::source_info source{body.c_str(), body.size()};
    auto options = cudf::io::json_reader_options::builder(source).lines(true);
    return cudf::io::read_json(options.build());
}

void RabbitMQSourceStage::close()
{
    // disconnect
    if (m_channel)
    {
        m_channel.reset();
    }
}
```

## Python Proxy & Interface

```cpp
std::shared_ptr<RabbitMQSourceStage>
RabbitMQSourceStageInterfaceProxy::init(neo::Segment &segment,
                                        const std::string &name,
                                        const std::string &host,
                                        const std::string &exchange,
                                        const std::string &exchange_type,
                                        const std::string &queue_name,
                                        std::chrono::milliseconds poll_interval)
{
    auto stage =
        std::make_shared<RabbitMQSourceStage>(segment, name, host, exchange, exchange_type, queue_name, poll_interval);
    segment.register_node<RabbitMQSourceStage>(stage);
    return stage;
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_rabbit, m)
{
    py::class_<RabbitMQSourceStage, neo::SegmentObject, std::shared_ptr<RabbitMQSourceStage>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("segment"),
             py::arg("name"),
             py::arg("host"),
             py::arg("exchange"),
             py::arg("exchange_type") = "fanout",
             py::arg("queue_name")    = "",
             py::arg("poll_interval") = 100ms);
}
```

Wrapping it all together our source file should look like:
`examples/rabbitmq/_lib/rabbitmq_source.cpp`

```cpp
#include "rabbitmq_source.hpp"

#include <neo/core/segment.hpp>
#include <neo/core/segment_object.hpp>

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>

#include <glog/logging.h>
#include <pybind11/chrono.h>  // for timedelta->chrono
#include <pybind11/pybind11.h>
#include <boost/fiber/operations.hpp>  // for this_fiber::sleep_for

#include <exception>
#include <sstream>
#include <vector>

namespace morpheus_rabbit {

RabbitMQSourceStage::RabbitMQSourceStage(const neo::Segment &segment,
                                         const std::string &name,
                                         const std::string &host,
                                         const std::string &exchange,
                                         const std::string &exchange_type,
                                         const std::string &queue_name,
                                         std::chrono::milliseconds poll_interval) :
  neo::SegmentObject(segment, name),
  base_t(segment, name, build_observable()),
  m_channel{AmqpClient::Channel::Create(host)},
  m_poll_interval{poll_interval}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}

neo::Observable<RabbitMQSourceStage::source_type_t> RabbitMQSourceStage::build_observable()
{
    return neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t> &subscriber) {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception &e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    });
}

void RabbitMQSourceStage::source_generator(neo::Subscriber<RabbitMQSourceStage::source_type_t> &subscriber)
{
    const std::string consumer_tag = m_channel->BasicConsume(m_queue_name, "", true, false);
    while (subscriber.is_subscribed())
    {
        AmqpClient::Envelope::ptr_t envelope;
        if (m_channel->BasicConsumeMessage(consumer_tag, envelope, 0))
        {
            try
            {
                auto table   = from_json(envelope->Message()->Body());
                auto message = MessageMeta::create_from_cpp(std::move(table), 0);
                subscriber.on_next(std::move(message));
            } catch (const std::exception &e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_channel->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            boost::this_fiber::sleep_for(m_poll_interval);
        }
    }
}

cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string &body) const
{
    cudf::io::source_info source{body.c_str(), body.size()};
    auto options = cudf::io::json_reader_options::builder(source).lines(true);
    return cudf::io::read_json(options.build());
}

void RabbitMQSourceStage::close()
{
    // disconnect
    if (m_channel)
    {
        m_channel.reset();
    }
}

// ************ WriteToFileStageInterfaceProxy ************* //
std::shared_ptr<RabbitMQSourceStage>
RabbitMQSourceStageInterfaceProxy::init(neo::Segment &segment,
                                        const std::string &name,
                                        const std::string &host,
                                        const std::string &exchange,
                                        const std::string &exchange_type,
                                        const std::string &queue_name,
                                        std::chrono::milliseconds poll_interval)
{
    auto stage =
        std::make_shared<RabbitMQSourceStage>(segment, name, host, exchange, exchange_type, queue_name, poll_interval);
    segment.register_node<RabbitMQSourceStage>(stage);
    return stage;
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_rabbit, m)
{
    py::class_<RabbitMQSourceStage, neo::SegmentObject, std::shared_ptr<RabbitMQSourceStage>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("segment"),
             py::arg("name"),
             py::arg("host"),
             py::arg("exchange"),
             py::arg("exchange_type") = "fanout",
             py::arg("queue_name")    = "",
             py::arg("poll_interval") = 100ms);
}

}  // namespace morpheus_rabbit
```

## Python Changes

Previously, our stage connected to the RabbitMQ server in the constructor. This is no longer advantageous to us when C++ execution is enabled; instead, we will record our constructor arguments and move the connection code to a new `connect` method. Our new constructor and `connect` methods are updated to:

```python
def __init__(self,
             config: Config,
             host: str,
             exchange: str,
             exchange_type: str='fanout',
             queue_name: str='',
             poll_interval: timedelta=None):
    super().__init__(config)
    self._host = host
    self._exchange = exchange
    self._exchange_type = exchange_type
    self._queue_name = queue_name

    self._connection = None
    self._channel = None

    if poll_interval is not None:
        self._poll_interval = poll_interval
    else:
        self._poll_interval = timedelta(milliseconds=100)

def connect(self):
    self._connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=self._host))

    self._channel = self._connection.channel()
    self._channel.exchange_declare(
        exchange=self._exchange, exchange_type=self._exchange_type)

    result = self._channel.queue_declare(
        queue=self._queue_name, exclusive=True)

    # When queue_name='' we will receive a randomly generated queue name
    self._queue_name = result.method.queue

    self._channel.queue_bind(
        exchange=self._exchange, queue=self._queue_name)
```

Lastly our `_build_source` method needs to be updated to build a C++ node when `morpheus.config.CppConfig` is configured to `True`.

```python
def _build_source(self, seg: neo.Segment) -> StreamPair:
    if CppConfig.get_should_use_cpp():
        node = morpheus_rabbit_cpp.RabbitMQSourceStage(
            seg,
            self.unique_name,
            self._host,
            self._exchange,
            self._exchange_type,
            self._queue_name,
            self._poll_interval
        )
    else:
        self.connect()
        node = seg.make_source(self.unique_name, self.source_generator)
    return node, MessageMeta
```
