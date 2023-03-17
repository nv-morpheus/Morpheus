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

# Creating a Python Morpheus Module

## Background

Morpheus makes use of the MRC graph-execution framework. Morpheus pipelines are built on top of MRC pipelines, which are
comprised of collections of nodes and edges, called segments (think sub-graphs), which can in turn be connected by
ingress/egress ports. In many common cases, an MRC pipeline will consist of only a single segment. While Morpheus
Stages are the primary building blocks of Morpheus pipelines, Morpheus Modules can be thought of as a way to define
basic units of work, which can, in turn, be composed and (re)used to build more complex stages. Modules can be
written in Python or C++.

## The Passthrough Module

The `passthrough` module is a simple module that takes a single input port and a single output port. It simply
passes it forward, in much the same way that the example stage defined in the [Simple Python Stage](.
/guides/1_simple_python_stage.md) does; however, it only defines actual unit of work, and must then be loaded either as
its own Morpheus stage, or within the context of another stage in order to be used.

### Module Definition and Registration

`my_test_module.py`

```python
import mrc
from mrc.core import operators as ops

from morpheus.utils.module_utils import register_module


@register_module("my_test_module", "my_module_namespace")
def my_test_module_initialization(builder: mrc.Builder):
    module_config = builder.get_current_module_config()  # Get the module configuration

    def on_data(data):
        return data

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("my_test_module_forwarding_node", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)
```

Here, we define a module, or rather a blueprint for creating a module, named `my_test_module` in the
`my_module_namespace` namespace. The `register_module` decorator is used to register the module with the system and
make it available to be loaded by other modules, stages, or pipelines. The `register_module` decorator takes two
parameters: the name of the module, and the namespace in which the module is defined. The namespace is used to avoid
naming collisions between core morpheus, custom, and third party modules.

The `my_test_module_initialization` function is called by the Morpheus module loader when the module is loaded. It
then creates a new instance of the module, which creates the appropriate MRC nodes and edges, and registers inputs
and outputs that other modules or MRC nodes can connect to.

Note that we also obtain a 'module_config' object from the builder. This object is a dictionary that contains all
configuration parameters that were passed to the module when it was loaded. This is useful for allowing modules to
customize their behavior based on runtime parameters. We will see an example of this in the next section.

### Loading the Module

After a module has been defined and registered, it can be loaded by other modules or stages. Below we
illustrate this process, in both cases: first usage within another module, and second we'll load the module we just
created as simple stage, a process that specializes the general behavior of the existing `LinearModuleStage`.

`my_test_module_consumer.py`

```python
@register_module("my_test_module_consumer", "my_module_namespace")
def my_test_module_consumer_initialization(builder: mrc.Builder):
    consumer_module_config = builder.get_current_module_config()  # Get the module configuration
    module_config = {
        "some_configuration_parameter": "some_value"
    }

    my_test_module = builder.load_module("my_test_module", "my_module_namespace", "module_instance_name", module_config)

    builder.register_module_input("input_0", my_test_module.input_port("input_0"))
    builder.register_module_output("output_0", my_test_module.output_port("output_0"))
```

Here, we've defined a new module that loads the `my_test_module` module that we defined above, and then connects
directly to its input and output ports. Obviously this is a trivial example, but it illustrates the basic process and
ease of use when loading and incorporating modules into existing workflows.

`my_test_module_consumer_stage.py`

```python
class MyPassthroughModuleWrapper(SinglePortStage):
    # ... stage implementation 
    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        module_config = {
            "some_configuration_parameter": "some_value"
        }

        my_module = builder.load_module("my_test_module", "my_module_namespace", "module_instance_name", module_config)

        module_in_stream = my_module.input_port("input_0")
        module_out_stream = my_module.output_port("output_0")

        builder.make_edge(input_stream[0], module_in_stream)

        return module_out_stream, self._output_type
```

Here, we've defined a new stage that loads the `my_test_module` module that we defined above, and then wraps its
input and output connections.

### Module Chaining and Nesting

Modules can be arbitrarily nested, and can be chained together to create more complex modules. For example, lets
define a slightly more interesting module that takes an integer input, `i`, and outputs `(i^2 + 3*i)`.
For this, we'll define three new modules, `my_square_module` and `my_times_three_module`, that perform the
appropriate operations, and `my_compound_op_module` which wraps them both. We'll then construct a single new module as a
composition of these three modules.

```python
import mrc
from mrc.core import operators as ops

from morpheus.utils.module_utils import register_module


## Create and register our two new component modules 
# ==========================================
@register_module("my_square_module", "my_module_namespace")
def my_test_module_initialization(builder: mrc.Builder):
    def on_data(data: int):
        return data ** 2

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("square", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)


@register_module("my_times_three_module", "my_module_namespace")
def my_test_module_initialization(builder: mrc.Builder):
    def on_data(data: int):
        return 3 * data

    def node_fn(observable: mrc.Observable, subscriber: mrc.Subscriber):
        observable.pipe(ops.map(on_data)).subscribe(subscriber)

    node = builder.make_node("times_two", mrc.core.operators.build(node_fn))

    builder.register_module_input("input_0", node)
    builder.register_module_output("output_0", node)


## Create and register our new compound operator -- illustrates module chaining
@register_module("my_compound_op_module", "my_module_namespace")
def my_compound_op(builder: mrc.Builder):
    square_module = builder.load_module("my_square_module", "my_module_namespace", "square_module")
    times_three_module = builder.load_module("my_times_three_module", "my_module_namespace", "times_three_module")

    builder.make_edge(square_module.output_port("output_0"), times_three_module.input_port("input_0"))

    builder.register_module_input("input_0", square_module.input_port("input_0"))
    builder.register_module_output("output_0", times_three_module.output_port("output_0"))


## Create and register our new compound module -- illustrates module nesting
@register_module("my_compound_module", "my_module_namespace")
def my_compound_module(builder: mrc.Builder):
    op_module = builder.load_module("my_compound_op_module", "my_module_namespace", "op_module")

    builder.register_module_input("input_0", op_module.input_port("input_0"))
    builder.register_module_output("output_0", op_module.output_port("output_0"))
```

`my_compound_module_consumer_stage.py`

```python
class MyCompoundOpModuleWrapper(SinglePortStage):
    # ... stage implementation 
    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        module_config = {}

        my_module = builder.load_module("my_compound_module", "my_module_namespace", "module_instance_name",
                                        module_config)

        module_in_stream = my_module.input_port("input_0")
        module_out_stream = my_module.output_port("output_0")

        builder.make_edge(input_stream[0], module_in_stream)

        return module_out_stream, self._output_type
```

### Wrapping Modules in Practice

While we have created new stages for our example modules here, in general we would not define an entirely new stage
just to wrap a module. Instead, we would use the `LinearModuleStage` to wrap the module:

```python
from morpheus.stages.general.linear_modules_stage import LinearModulesStage

config = Config()  # Morpheus config
module_config = {
    "module_id": "my_compound_module",
    "module_namespace": "my_module_namespace",
    "module_instance_name": "module_instance_name",
    ... other module config params...
}

pipeline.add_stage(LinearModulesStage(config, module_config, input_port_name="input_0", output_port_name="output_0"))
```