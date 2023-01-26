<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Creating a C++ Source Stage

For this example, we are going to add a C++ implementation for the `RabbitMQSourceStage` we designed in the Python examples. The Python implementation of this stage emits messages of the type `MessageMeta`; as such, our C++ implementation must do the same.

For communicating with [RabbitMQ](https://www.rabbitmq.com/) we will be using the [SimpleAmqpClient](https://github.com/alanxz/SimpleAmqpClient) library, and [libcudf](https://docs.rapids.ai/api/libcudf/stable/index.html) for constructing the `DataFrame`.

## Header Definition

Our includes:

```cpp
#include <SimpleAmqpClient/SimpleAmqpClient.h>  // for AmqpClient::Channel::ptr_t
#include <cudf/io/types.hpp>                    // for cudf::io::table_with_metadata
#include <morpheus/messages/meta.hpp>           // for MessageMeta
#include <mrc/segment/builder.hpp>              // for Segment Builder
#include <mrc/segment/object.hpp>               // for Segment Object
#include <pymrc/node.hpp>                       // for mrc::pymrc::PythonSource

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>
```

The MRC includes bring in the definitions for MRC `Builder`, `SegmentObject` and `PythonSource`.

Our namespace and class definition is:

```cpp
namespace morpheus_rabbit {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace std::literals;
using namespace morpheus;

class RabbitMQSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;
```

Our base class defines `source_type_t` as an alias for `std::shared_ptr<MessageMeta>` which we are going to use as it will occur in some of our function signatures. The way to think about `source_type_t` is that the stage we are writing emits objects of type `MessageMeta`. The `subscriber_fn_t` is an alias for a function which will receive an `rxcpp::subscriber` instance and emit messages into the pipeline.  The class we are deriving from, `PythonSource`, defines both of these to make writing function signatures easier.

Our constructor is similar to the constructor of our Python class with the majority of the parameters being specific to communicating with RabbitMQ. In this case the default destructor is sufficient.

```cpp
RabbitMQSourceStage(const std::string& host,
                    const std::string& exchange,
                    const std::string& exchange_type        = "fanout"s,
                    const std::string& queue_name           = ""s,
                    std::chrono::milliseconds poll_interval = 100ms);
~RabbitMQSourceStage() override = default;
```

Our class will require a few private methods

```cpp
subscriber_fn_t build();
void source_generator(rxcpp::subscriber<source_type_t> subscriber);
cudf::io::table_with_metadata from_json(const std::string& body) const;
void close();
```

The `build` method is responsible for returning a function with a signature matching `subscriber_fn_t`, the result of which will be passed into our base's constructor. Typically, this function is the center of a source stage, making calls to the `subscriber`'s `on_next`, `on_error`, and `on_completed` methods. For this example, the RabbitMQ-specific logic was broken out into the `source_generator` method, which should be analogous to the `source_generator` method from the Python class, and will emit new messages into the pipeline by calling `subscriber.on_next(message)`.

The `from_json` method parses a JSON string to a cuDF [table_with_metadata](https://docs.rapids.ai/api/libcudf/stable/structcudf_1_1io_1_1table__with__metadata.html). Lastly, the `close` method disconnects from the RabbitMQ exchange.

We will also need three private attributes specific to our interactions with RabbitMQ: our polling interval, the name of the queue we are listening to, and a pointer to our channel object.

```cpp
std::chrono::milliseconds m_poll_interval;
std::string m_queue_name;
AmqpClient::Channel::ptr_t m_channel;
```

Wrapping it all together, our header file should be similar to:
```cpp
#pragma once

#include <SimpleAmqpClient/SimpleAmqpClient.h>  // for AmqpClient::Channel::ptr_t
#include <cudf/io/types.hpp>                    // for cudf::io::table_with_metadata
#include <morpheus/messages/meta.hpp>           // for MessageMeta
#include <mrc/segment/builder.hpp>              // for Segment Builder
#include <mrc/segment/object.hpp>               // for Segment Object
#include <pymrc/node.hpp>                       // for mrc::pymrc::PythonSource

#include <chrono>  // for chrono::milliseconds
#include <memory>  // for shared_ptr
#include <string>

namespace morpheus_rabbit {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace std::literals;
using namespace morpheus;

class RabbitMQSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    RabbitMQSourceStage(const std::string& host,
                        const std::string& exchange,
                        const std::string& exchange_type        = "fanout"s,
                        const std::string& queue_name           = ""s,
                        std::chrono::milliseconds poll_interval = 100ms);

    ~RabbitMQSourceStage() override = default;

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);
    cudf::io::table_with_metadata from_json(const std::string& body) const;
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
    static std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>> init(mrc::segment::Builder& builder,
                                                                           const std::string& name,
                                                                           const std::string& host,
                                                                           const std::string& exchange,
                                                                           const std::string& exchange_type,
                                                                           const std::string& queue_name,
                                                                           std::chrono::milliseconds poll_interval);
};
#pragma GCC visibility pop
}  // namespace morpheus_rabbit
```

## Source Definition

Our includes section is:

```cpp
#include "rabbitmq_source.hpp"

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <glog/logging.h>
#include <pybind11/chrono.h>  // for timedelta->chrono conversions
#include <pybind11/pybind11.h>

#include <exception>
#include <sstream>
#include <thread>  // for std::this_thread::sleep_for
#include <vector>
```

The [Google Logging Library](https://github.com/google/glog) (glog) is used internally by Morpheus for logging; however, the choice of a logger is up to the individual developer.

Authors of stages that require concurrency are free to choose their own concurrency models. Launching new processes, threads, or fibers from within a stage is permissible as long as the management of those resources is contained within the stage. Newly launched threads are also free to use thread-local storage so long as it doesn't occur within the thread the stage is executed from.

The definition for our constructor is:

```cpp
RabbitMQSourceStage::RabbitMQSourceStage(const std::string& host,
                                         const std::string& exchange,
                                         const std::string& exchange_type,
                                         const std::string& queue_name,
                                         std::chrono::milliseconds poll_interval) :
  PythonSource(build()),
  m_channel{AmqpClient::Channel::Create(host)},
  m_poll_interval{poll_interval}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}
```

The key thing to note is the invocation of our base's constructor is the result of the `build` method:

```cpp
PythonSource(build()),
```

Our `build` method returns a function, which needs to do three things:
1. Emit data into the pipeline by calling `rxcpp::subscriber`'s `on_next` method. In our example, this occurs in the `source_generator` method.
1. When an error occurs, call the `rxcpp::subscriber`'s `on_error` method.
1. When we are done, call the `rxcpp::subscriber`'s `on_complete` method.

Note: For some source stages, such as ones that read input data from a file, there is a clear point where the stage is complete. Others such as this one are intended to continue running until it is shut down. For the latter situation, the stage can poll the `rxcpp::subscriber`'s `is_subscribed` method, which will return a value of `false` on shut down.

```cpp
RabbitMQSourceStage::subscriber_fn_t RabbitMQSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> subscriber) -> void {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    };
}
```

As a design decision, we left the majority of the RabbitMQ specific code in the `source_generator` method, leaving Morpheus specific code in the `build` method. For each message that we receive and can successfully parse, we call the `rxcpp::subscriber`'s `on_next` method. When there are no messages in the queue, we yield the thread by sleeping, potentially allowing the scheduler to perform a context switch.

```cpp
void RabbitMQSourceStage::source_generator(rxcpp::subscriber<RabbitMQSourceStage::source_type_t> subscriber)
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
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_channel->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            std::this_thread::sleep_for(m_poll_interval);
        }
    }
}
```

## A note on performance:

We don't yet know the size of the messages that we are going to receive from RabbitMQ, but we should assume that they may be quite large. As such, we try to limit the number of copies of this data, preferring to instead pass by reference or move data. The `SimpleAmqpClient`'s `Body()` method returns a const reference to the payload, which we also pass by reference into the `from_json` method. Since our stage has no need for the data itself after it's emitted into the pipeline, we move our cuDF data table when we construct our `MessageMeta` instance, and then we once again move the message into the subscriber's `on_next` method.

Our `from_json` and `close` methods are rather straightforward:

```cpp
cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string& body) const
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
std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>> RabbitMQSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::string& host,
    const std::string& exchange,
    const std::string& exchange_type,
    const std::string& queue_name,
    std::chrono::milliseconds poll_interval)
{
    return builder.construct_object<RabbitMQSourceStage>(
        name, host, exchange, exchange_type, queue_name, poll_interval);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_rabbit, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<RabbitMQSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("host"),
             py::arg("exchange"),
             py::arg("exchange_type") = "fanout",
             py::arg("queue_name")    = "",
             py::arg("poll_interval") = 100ms);
}
```

Wrapping it all together, our source file should be:
```cpp
#include "rabbitmq_source.hpp"

#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <glog/logging.h>
#include <pybind11/chrono.h>  // for timedelta->chrono conversions
#include <pybind11/pybind11.h>

#include <exception>
#include <sstream>
#include <thread>  // for std::this_thread::sleep_for
#include <vector>

namespace morpheus_rabbit {

RabbitMQSourceStage::RabbitMQSourceStage(const std::string& host,
                                         const std::string& exchange,
                                         const std::string& exchange_type,
                                         const std::string& queue_name,
                                         std::chrono::milliseconds poll_interval) :
  PythonSource(build()),
  m_channel{AmqpClient::Channel::Create(host)},
  m_poll_interval{poll_interval}
{
    m_channel->DeclareExchange(exchange, exchange_type);
    m_queue_name = m_channel->DeclareQueue(queue_name);
    m_channel->BindQueue(m_queue_name, exchange);
}

RabbitMQSourceStage::subscriber_fn_t RabbitMQSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> subscriber) -> void {
        try
        {
            this->source_generator(subscriber);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Encountered error while polling RabbitMQ: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            close();
            return;
        }

        close();
        subscriber.on_completed();
    };
}

void RabbitMQSourceStage::source_generator(rxcpp::subscriber<RabbitMQSourceStage::source_type_t> subscriber)
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
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Error occurred converting RabbitMQ message to Dataframe: " << e.what();
            }
            m_channel->BasicAck(envelope);
        }
        else
        {
            // Sleep when there are no messages
            std::this_thread::sleep_for(m_poll_interval);
        }
    }
}

cudf::io::table_with_metadata RabbitMQSourceStage::from_json(const std::string& body) const
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

std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>> RabbitMQSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::string& host,
    const std::string& exchange,
    const std::string& exchange_type,
    const std::string& queue_name,
    std::chrono::milliseconds poll_interval)
{
    return builder.construct_object<RabbitMQSourceStage>(
        name, host, exchange, exchange_type, queue_name, poll_interval);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_rabbit, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<RabbitMQSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<RabbitMQSourceStage>>>(
        m, "RabbitMQSourceStage", py::multiple_inheritance())
        .def(py::init<>(&RabbitMQSourceStageInterfaceProxy::init),
             py::arg("builder"),
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

Previously, our stage connected to the RabbitMQ server in the constructor. This is no longer advantageous to us when C++ execution is enabled. Instead, we will record our constructor arguments and move the connection code to a new `connect` method. Our new constructor and `connect` methods are updated to:

```python
def __init__(self,
             config: Config,
             host: str,
             exchange: str,
             exchange_type: str = 'fanout',
             queue_name: str = '',
             poll_interval: str = '100millis'):
    super().__init__(config)
    self._host = host
    self._exchange = exchange
    self._exchange_type = exchange_type
    self._queue_name = queue_name

    self._connection = None
    self._channel = None

    self._poll_interval = pd.Timedelta(poll_interval)

    # Flag to indicate whether or not we should stop
    self._stop_requested = False
```
```python
def connect(self):
    self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=self._host))

    self._channel = self._connection.channel()
    self._channel.exchange_declare(exchange=self._exchange, exchange_type=self._exchange_type)

    result = self._channel.queue_declare(queue=self._queue_name, exclusive=True)

    # When queue_name='' we will receive a randomly generated queue name
    self._queue_name = result.method.queue

    self._channel.queue_bind(exchange=self._exchange, queue=self._queue_name)
```

Lastly, our `_build_source` method needs to be updated to build a C++ node when `morpheus.config.CppConfig.get_should_use_cpp()` is configured to `True` by using the `self._build_cpp_node()` method.

```python
def _build_source(self, builder: mrc.Builder) -> StreamPair:
    if self._build_cpp_node():
        node = morpheus_rabbit_cpp.RabbitMQSourceStage(builder,
                                                        self.unique_name,
                                                        self._host,
                                                        self._exchange,
                                                        self._exchange_type,
                                                        self._queue_name,
                                                        self._poll_interval.to_pytimedelta())
    else:
        self.connect()
        node = builder.make_source(self.unique_name, self.source_generator)
    return node, MessageMeta
```
