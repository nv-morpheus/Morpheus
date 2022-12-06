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

# 3. A Simple C++ Stage

Morpheus offers the choice of writing pipeline stages in either Python or C++. For many use cases, a Python stage is perfectly fine. However, in the event that a Python stage becomes a bottleneck for the pipeline, then writing a C++ implementation for the stage becomes advantageous. The C++ implementations of Morpheus stages and messages utilize the [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) library to provide Python bindings.

So far we have been defining our pipelines in Python. Most of the stages included with Morpheus have both a Python and a C++ implementation, and Morpheus will use the C++ implementations by default. You can explicitly disable the use of C++ stage implementations by calling `morpheus.config.CppConfig.set_should_use_cpp(False)`:

```python
from morpheus.config import CppConfig
CppConfig.set_should_use_cpp(False)
```

If a stage does not have a C++ implementation, Morpheus will fall back to the Python implementation without any additional configuration and operate in a hybrid execution mode.

In addition to C++ accelerated stage implementations, Morpheus also provides a C++ implementation for message primitives. When C++ execution is enabled, constructing one of the Python message classes defined under `morpheus.messages` will return a Python object with bindings to the underlying C++ implementation.

Since we are defining our pipelines in Python, it becomes the responsibility of the Python implementation to build a C++ accelerated node. This happens in the `_build_source` and `_build_single` methods. Ultimately it is the decision of a Python stage to build a Python node or a C++ node. It is perfectly acceptable to build a Python node when `morpheus.config.CppConfig.get_should_use_cpp()` is configured to `True`. It is not acceptable, however, to build a C++ node when `morpheus.config.CppConfig.get_should_use_cpp() == False`. The reason is the C++ implementations of Morpheus' messages can be consumed by Python and C++ stage implementations alike. However when `morpheus.config.CppConfig.get_should_use_cpp() == False`, the Python implementations of each message type will be used which cannot be consumed by the C++ implementations of stages.

Python stages which have a C++ implementation must advertise this functionality by returning a value of `True` from the `supports_cpp_node` method:

```python
def supports_cpp_node(self):
    return True
```

C++ message object declarations can be found in the header files that are located in the `morpheus/_lib/include/morpheus/messages` directory. For example, the `MessageMeta` class declaration is located in `morpheus/_lib/include/morpheus/messages/meta.hpp`. In code this would be included as:

```cpp
#include <morpheus/messages/meta.hpp>
```

Morpheus C++ source stages inherit from the `PythonSource` class:

```cpp
template <typename SourceT>
class PythonSource : ...
```

The `SourceT` type will be the datatype emitted by this stage. In contrast, general stages and sinks must inherit from the `PythonNode` class, which specifies both receive and emit types:

```cpp
template <typename SinkT, typename SourceT = SinkT>  // by default, emit type == receive type
class PythonNode : ...
```

Both the `PythonSource` and `PythonNode` classes are defined in the `pysrf/node.hpp` header.

Note: `SourceT` and `SinkT` types are typically `shared_ptr`s to a Morpheus message type. For example, `std::shared_ptr<MessageMeta>`. This allows the reference counting mechanisms used in Python and C++ to share the same count, properly cleaning up the objects when they are no longer referenced.

Note: The C++ implementation of a stage must receive and emit the same message types as the Python implementation.

Note: The "Python" in the `PythonSource` & `PythonNode` class names refers to the fact that these classes read and write objects registered with python, not the implementation language.

## A Simple Pass Through Stage

As in our Python guide, we will start with a simple pass through stage which can be used as a starting point for future development of other stages. Note that by convention, C++ classes in Morpheus have the same name as their corresponding Python classes and are located under a directory named `_lib`. We will be following that convention. To start, we will create a `_lib` directory and a new empty `__init__.py` file.

While our Python implementation accepts messages of any type (in the form of Python objects), on the C++ side we don't have that flexibility since our node is subject to C++ static typing rules. In practice, this isn't a limitation as we usually know which specific message types we need to work with.

To start with, we have our Morpheus and SRF-specific includes:

```cpp
#include <morpheus/messages/multi.hpp>  // for MultiMessage
#include <pysrf/node.hpp>               // for PythonNode
#include <srf/segment/builder.hpp>      // for Segment Builder
#include <srf/segment/object.hpp>       // for Segment Object
```

We'll want to define our stage in its own namespace. In this case, we will name it `morpheus_example`, giving us a namespace and class definition like:

```cpp
namespace morpheus_example {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace morpheus;

class PassThruStage : public srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};
```

We explicitly set the visibility for the stage object in the namespace to default. This is due to a pybind11 requirement for module implementations to default symbol visibility to hidden (`-fvisibility=hidden`). More details about this can be found in the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes).

For simplicity, we defined `base_t` as an alias for our base class type because the definition can be quite long. Our base class type also defines a few additional type aliases for us: `subscribe_fn_t`, `sink_type_t` and `source_type_t`. The `sink_type_t` and `source_type_t` aliases are shortcuts for the sink and source types that this stage will be reading and writing. In this case both the `sink_type_t` and `source_type_t` resolve to `std::shared_ptr<MultiMessage>`. `subscribe_fn_t` (read as "subscribe function type") is an alias for:

```cpp
std::function<rxcpp::subscription(rxcpp::observable<InputT>, rxcpp::subscriber<OutputT>)>
```

This means that a SRF subscribe function accepts an `rxcpp::observable` of type `InputT` and `rxcpp::subscriber` of type `OutputT` and returns a subscription. In our case, both `InputT` and `OutputT` are `std::shared_ptr<MultiMessage>`.

All Morpheus C++ stages receive an instance of a SRF Segment Builder and a name (Typically this is the Python class' `unique_name` property) when constructed from Python. Note that C++ segments don't receive an instance of the Morpheus config. Therefore, if there are any attributes in the config needed by the C++ class, it is the responsibility of the Python class to extract them and pass them in as parameters to the C++ class.

We will also define an interface proxy object to keep the class definition separated from the Python interface. This isn't strictly required, but it is a convention used internally by Morpheus. Our proxy object will define a static method named `init` which is responsible for constructing a `PassThruStage` instance and returning it wrapped in a `shared_ptr`. There are many common Python types that pybind11 [automatically converts](https://pybind11.readthedocs.io/en/latest/advanced/cast/overview.html#conversion-table) to their associated C++ types. The SRF `Builder` is a C++ object with Python bindings. The proxy interface object is used to help insulate Python bindings from internal implementation details.

```cpp
struct PassThruStageInterfaceProxy
{
    static std::shared_ptr<srf::segment::Object<PassThruStage>> init(srf::segment::Builder &builder,
                                                                     const std::string &name);
};
```

### The Complete C++ Stage Header

Putting it all together, our header file looks like this:

```cpp
#pragma once

#include <morpheus/messages/multi.hpp>  // for MultiMessage
#include <pysrf/node.hpp>               // for PythonNode
#include <srf/segment/builder.hpp>      // for Segment Builder
#include <srf/segment/object.hpp>       // for Segment Object

#include <memory>
#include <string>

namespace morpheus_example {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace morpheus;

class PassThruStage : public srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};

struct PassThruStageInterfaceProxy
{
    static std::shared_ptr<srf::segment::Object<PassThruStage>> init(srf::segment::Builder &builder,
                                                                     const std::string &name);
};

#pragma GCC visibility pop
}  // namespace morpheus_example
```

### Source Code Definition

Our includes section looks like:

```cpp
#include "pass_thru.hpp"

#include <pybind11/pybind11.h>
#include <pysrf/utils.hpp>  // for pysrf::import

#include <exception>
```

The constructor for our class is responsible for passing the output of `build_operator` to our base class, as well as calling the constructor for `PythonNode`:

```cpp
PassThruStage::PassThruStage() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}
```

Note that the output of `build_operator()` is not passed directly to the `PythonNode` constructor and instead gets passed to `base_t::op_factory_from_sub_fn()`. This is because reactive operators can be defined two ways:

1. Using the short form `std::function<rxcpp::observable<T>(rxcpp::observable<R>)` which is good when you can use an existing `rxcpp` operator
2. Using the long form `std::function<rxcpp::subscription(rxcpp::observable<T>, rxcpp::subscriber<R>)>` which allows for more customization and better control over the lifetime of objects.

It's possible to convert between the two signatures which is exactly what `base_t::op_factory_from_sub_fn()` does. If you wanted to use the short form, you could define the constructor of `PassThruStage` using:

```cpp
PassThruStage::PassThruStage() :
  PythonNode([](rxcpp::observable<sink_type_t> obs){ return obs; })
{}
```

However, this doesnt illustrate well how to customize a stage. So we will be using the long form signature for our examples.

The `build_operator` method defines an observer who is subscribed to our input `rxcpp::observable`. The observer consists of three functions that are typically lambdas:  `on_next`, `on_error`, and `on_completed`. Typically, these three functions call the associated methods on the output subscriber.

```cpp
PassThruStage::subscribe_fn_t PassThruStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(
            rxcpp::make_observer<sink_type_t>([this, &output](sink_type_t x) { output.on_next(std::move(x)); },
                                              [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                                              [&]() { output.on_completed(); }));
    };
}
```

Note the use of `std::move` in the `on_next` function. In Morpheus, our messages often contain both large payloads as well as Python objects where performing a copy necessitates acquiring the Python [Global Interpreter Lock (GIL)](https://docs.python.org/3/glossary.html#term-global-interpreter-lock). In either case, unnecessary copies can become a performance bottleneck, and much care is taken to limit the number of copies required for data to move through the pipeline.

There are situations in which a C++ stage does need to interact with Python, and therefore acquiring the GIL is a requirement. In these situations, it is important to ensure that the GIL is released before calling the `on_next` method. This is typically accomplished using pybind11's [gil_scoped_acquire](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil) RAII class inside of a code block. Consider the following `on_next` lambda function from Morpheus' `SerializeStage`:

```cpp
[this, &output](sink_type_t msg) {
    auto table_info = this->get_meta(msg);
    std::shared_ptr<MessageMeta> meta;
    {
        pybind11::gil_scoped_acquire gil;
        meta = MessageMeta::create_from_python(std::move(table_info.as_py_object()));
    } // GIL is released
    output.on_next(std::move(meta));
}
```

We scoped the acquisition of the GIL such that it is held only for the parts of the code where it is strictly necessary. In the above example, when we exit the code block, the `gil` variable will go out of scope and release the global interpreter lock.

### Python Proxy and Interface

The things that all proxy interfaces need to do are:
1. Construct the stage using the `srf::segment::Builder::construct_object` method
2. Return a `shared_ptr` to the stage wrapped in a `srf::segment::Object`

```cpp
std::shared_ptr<srf::segment::Object<PassThruStage>> PassThruStageInterfaceProxy::init(srf::segment::Builder& builder,
                                                                                       const std::string& name)
{
    return builder.construct_object<PassThruStage>(name);
}
```

The Python interface itself defines a Python module named `morpheus_example` and a Python class in that module named `PassThruStage`. Note that the only method we are exposing to Python is the constructor. The class will be exposed to Python code as `lib_.morpheus_example.PassThruStage`.

```cpp
namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_example, m)
{
    srf::pysrf::import(m, "morpheus._lib.messages");

    py::class_<srf::segment::Object<PassThruStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PassThruStage>>>(m, "PassThruStage", py::multiple_inheritance())
        .def(py::init<>(&PassThruStageInterfaceProxy::init), py::arg("builder"), py::arg("name"));
}
```

### The Complete C++ Stage Implementation

```cpp
#include "pass_thru.hpp"

#include <pybind11/pybind11.h>
#include <pysrf/utils.hpp>  // for pysrf::import

#include <exception>

namespace morpheus_example {

PassThruStage::PassThruStage() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}

PassThruStage::subscribe_fn_t PassThruStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(
            rxcpp::make_observer<sink_type_t>([this, &output](sink_type_t x) { output.on_next(std::move(x)); },
                                              [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                                              [&]() { output.on_completed(); }));
    };
}

std::shared_ptr<srf::segment::Object<PassThruStage>> PassThruStageInterfaceProxy::init(srf::segment::Builder& builder,
                                                                                       const std::string& name)
{
    return builder.construct_object<PassThruStage>(name);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(morpheus_example, m)
{
    srf::pysrf::import(m, "morpheus._lib.messages");

    py::class_<srf::segment::Object<PassThruStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PassThruStage>>>(m, "PassThruStage", py::multiple_inheritance())
        .def(py::init<>(&PassThruStageInterfaceProxy::init), py::arg("builder"), py::arg("name"));
}

}  // namespace morpheus_example
```

### Python Changes

We need to make a few minor adjustments to our Python implementation of the `PassThruStage`. First, we import new `morpheus_example` Python module we created in the previous section.

```python
from _lib import morpheus_example as morpheus_example_cpp
```

As mentioned in the previous section, we will need to change the return value of `supports_cpp_node` to indicate that our stage supports a C++ implementation.  Our `_build_single` method needs to be updated to build a C++ node when `morpheus.config.CppConfig.get_should_use_cpp()` is `True` using the `self._build_cpp_node()` method. The `_build_cpp_node()` method compares both `morpheus.config.CppConfig.get_should_use_cpp()` and `supports_cpp_node()` and returns `True` only when both methods return `True`.

```python
def supports_cpp_node(self):
   return True
```
```python
def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
    if self._build_cpp_node():
        print("building C++ node")
        node = morpheus_example_cpp.PassThruStage(builder, self.unique_name)
    else:
        node = builder.make_node(self.unique_name, self.on_data)

    builder.make_edge(input_stream[0], node)
    return node, input_stream[1]
```
