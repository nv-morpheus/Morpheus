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

# 1. A Simple Python Stage

## Background

Morpheus makes use of the Neo graph-execution framework. Morpheus pipelines are built on top of Neo pipelines. Pipelines in Neo are made up of segments; however, in many common cases, a Neo pipeline will consist of only a single segment. Our Morpheus stages will interact with the Neo segment to build nodes and add them to the Neo graph. In the common case, a Morpheus stage will add a single node to the graph, but in some cases will add multiple nodes to the graph.

## The Pass Through Stage

To start, we will implement a single stage that could be included in a pipeline. For illustration, this stage will do nothing but take the input from the previous stage and forward it to the next stage. All Morpheus stages have several things in common, so while this doesn't do too much, it ends up being a good starting point for writing a new stage. From there, we can add our functionality as needed.

Defining this stage requires us to specify the stage type. Morpheus stages contain a single input and a single output inherited from `SinglePortStage`.  Stages that act as sources of data, in that they do not take an input from a prior stage but rather produce data from a source such as a file, Kafka service, or other external sources, will need to inherit from the `SingleOutputSource` base class.  We start our class definition with a few basic imports:

```python
import typing

import neo
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

class PassThruStage(SinglePortStage):
```

There are three methods that need to be defined in our new subclass to implement the stage interface: `name`, `accepted_types`, and `_build_single`. In practice, it is often necessary to define at least one more method which will perform the actual work of the stage; by convention, this method is typically named on_data, which we will define in our examples.

`name` is a property method; it should return a user-friendly name for the stage. Currently, this property is only used for debugging purposes, and there are no requirements on the content or format of the name.
```python
    @property
    def name(self) -> str:
        return "pass-thru"
```

The `accepted_types` method returns a tuple of message classes that this stage accepts as valid inputs. Morpheus uses this to validate that the parent of this stage emits a message that this stage can accept. Since our stage is a pass through, we will declare that we can accept any incoming message type; however, in practice, production stages will often declare a single Morpheus message class such as `MessageMeta` and `MultiMessage` (see the message classes defined in `morpheus.pipeline.messages` for a complete list).
```python
    def accepted_types(self) -> typing.Tuple:
        return (typing.Any,)
```

Our `on_data` method accepts the incoming message and returns a message. The returned message can be the same message instance that we received as our input or it could be a new message instance. The method is named `on_data` by convention; however, it is not part of the API. In the next section, we will register it as a callback in Morpheus.
```python
    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message
```

Finally, the `_build_single` method will be used at build time to wire our stage into the pipeline. `_build_single` receives an instance of the Neo pipeline segment along with a `StreamPair` instance. We will be using the segment instance to build a node from our stage and add it to the pipeline segment. The `StreamPair` argument is a tuple, the first element is our parent node, and the second is our parent node's output type. The return type of this method is also a `StreamPair`; typically, we will be returning our newly constructed node along with our output type.

In most cases, a Morpheus stage will define and build a single Neo node. In some advanced cases, a stage can construct more than one node. For our purposes, a Morpheus stage defines information about the type of node(s) it builds, while the node is the instance wired into the Neo pipeline. To build the node, we will call the `make_node` method of the segment instance, passing to it our name and our `on_data` method. We used the `unique_name` property, which will take the name property which we already defined and append a unique id to it.
```python
node = seg.make_node(self.unique_name, self.on_data)
```

Next, we will define an edge connecting our new node to our parent node:
```python
seg.make_edge(input_stream[0], node)
```

Finally, we will return a new tuple of our node and output type. Since this is a pass through node that can accept any input type, we will return our parent's type.
```python
return node, input_stream[1]
```

## Putting the Stage Together
```python
import typing

import neo

from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

class PassThruStage(SinglePortStage):
    @property
    def name(self) -> str:
        return "pass-thru"

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any,)

    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        node = seg.make_node(self.unique_name, self.on_data)
        seg.make_edge(input_stream[0], node)

        return node, input_stream[1]
```

## Testing our new Stage
To start testing our new pass through stage, we are going to construct a simple pipeline and add our new stage to it. This pipeline will do the minimum work necessary to verify our pass through stage: define a source stage, which will produce data and inject it into the pipeline; this data will be read and processed by our pass through stage, and then forwarded on to a monitoring stage which will record the messages it receives and terminate the pipeline.

Imports are needed for this example. This assumes we saved the code for the PassThruStage in a file named "pass_thru.py" in the same directory as this script which we will name "run_passthru.py".
```python
import logging
import os

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.utils.logging import configure_logging

from pass_thru import PassThruStage
```

Before constructing the pipeline, we need to do a bit of environment configuration, starting with the Morpheus logger:
```python
configure_logging(log_level=logging.DEBUG)
```

Next, we will build a Morpheus config object. In the next example, we will cover setting some common configuration parameters. For now, it is important to know that we will always need to build a config object.
```python
config = Config()
```

For our test, we will use the `FileSourceStage` to read in a large file containing lines of JSON corresponding to email messages and package them as Morpheus message objects for our pass through stage to consume.
```python
pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))
```

Next, we will add our new stage to the pipeline and then add a MonitorStage which will measure the throughput of our pass through stage.

```python
pipeline.add_stage(PassThruStage(config))
pipeline.add_stage(MonitorStage(config))
```

Finally, we run the pipeline:
```python
pipeline.run()
```

The output should look like this:
```
====Registering Pipeline====
====Registering Pipeline Complete!====
====Starting Pipeline====
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=./examples/data/email.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True, cudf_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <pass-thru-1; PassThruStage(args=(), kwargs={})>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <monitor-2; MonitorStage(description=Progress, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Pipeline Complete!====
Starting! Time: 1648834587.3092508
====Pipeline Started====
Progress[Complete]: 25229messages [00:00, 57695.02messages/s]
====Pipeline Complete====
```

## Putting the Pipeline Together
Note this code assumes that the `MORPHEUS_ROOT` environment variable is set to the root of the Morpheus project repository:
```python
import logging
import os

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.utils.logging import configure_logging

from pass_thru import PassThruStage

def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email.jsonlines')

   config = Config()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add our own stage
    pipeline.add_stage(PassThruStage(config))

    # Add monitor to record the performance of our new stage
    pipeline.add_stage(MonitorStage(config))

    # Run the pipeline
    pipeline.run()

if __name__ == "__main__":
    run_pipeline()
```
