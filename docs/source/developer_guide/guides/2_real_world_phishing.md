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

# 2. A Real-World Application: Phishing Detection

## Data Preprocessing

Now that we've seen a basic example of how to create a stage and use it in the context of a pipeline, we'll move on to a more advanced example that is representative of what we might want to do in a real-world situation. Given a set of records, each of which represents an email, suppose we want to predict which records correspond to fraudulent emails.

As part of this process, we might want to use a classification model trained on various pieces of metadata, such as recipient count, in addition to the raw content of each email. If we suppose this is true for our example, we need to build and connect a pre-processing stage to attach this information to each record before applying our classifier.

For this task, we'll need to define a new stage, which we will call our `RecipientFeaturesStage`, which will receive an input corresponding to an email, count the number of recipients in the email's metadata, and construct a Morpheus `MessageMeta` object that will contain the record content along with the augmented metadata.

For this stage, the code will look very similar to the previous example with a few notable changes. We will be working with the `MessageMeta` class. This is a Morpheus message containing a [cuDF](https://docs.rapids.ai/api/cudf/stable/) [DataFrame](https://docs.rapids.ai/api/cudf/stable/api_docs/dataframe.html); since we will expect our new stage to operate on {py:obj}`~morpheus.pipeline.messages.MultiMessage` types our new accepted_types method now looks like:

```python
def accepted_types(self) -> typing.Tuple:
    return (MessageMeta,)
```

Next, we will update our on_data method to perform the actual work.
We grab a reference to the incoming message's `df` attribute. It is important to note that this is a reference, and any changes made to this will be performed in place on the existing message instance.

```python
def on_data(self, message: MessageMeta):
    # Get the DataFrame from the incoming message
    df = message.df

    df['to_count'] = df['To'].str.count('@')
    df['bcc_count'] = df['BCC'].str.count('@')
    df['cc_count'] = df['CC'].str.count('@')
    df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

    # Attach features to string data
    df['data'] = (df['to_count'].astype(str) + '[SEP]' +
                    df['bcc_count'].astype(str) + '[SEP]' +
                    df['cc_count'].astype(str) + '[SEP]' +
                    df['total_recipients'].astype(str) + '[SEP]' +
                    df['Message'])

    # Return the message for the next stage
    return message
```

If mutating the data frame is undesirable, we could make a call to the data frame's [copy](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.copy.html#cudf.DataFrame.copy) method, and return a new message, note that this would come at the cost of performance and increased memory. Our updated `on_data` method would look like this (changing the first & last lines of the method):

```python
def on_data(self, message: MessageMeta):
    # Take a copy of the DataFrame from the incoming message
    df = message.df.copy(True)
    ...
    # Construct and return a new message containing our DataFrame
    return MessageMeta(df=df)
```


Our `_build_single` method remains unchanged; even though we are modifying the incoming messages, our input and output types remain the same.

## The Completed Preprocessing Stage

```python
import typing

import neo

from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair


class RecipientFeaturesStage(SinglePortStage):
    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta,)

    def on_data(self, message: MessageMeta):
        # Get the DataFrame from the incoming message
        df = message.df

        df['to_count'] = df['To'].str.count('@')
        df['bcc_count'] = df['BCC'].str.count('@')
        df['cc_count'] = df['CC'].str.count('@')
        df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

        # Attach features to string data
        df['data'] = (df['to_count'].astype(str) + '[SEP]' +
                      df['bcc_count'].astype(str) + '[SEP]' +
                      df['cc_count'].astype(str) + '[SEP]' +
                      df['total_recipients'].astype(str) + '[SEP]' +
                      df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        node = seg.make_node(self.unique_name, self.on_data)
        seg.make_edge(input_stream[0], node)

        return node, input_stream[1]
```

## Testing our new Stage

To ensure our new stage is working, we will use the pipeline from the previous example and insert our new stage where the pass through stage was previously, between the `FileSource` and  `MonitorStage` stages.

Update these lines:

```python
# Add our own stage
pipeline.add_stage(PassThruStage(config))

# Add monitor to record the performance of our new stage
pipeline.add_stage(MonitorStage(config))
```

To:

```python
# Add our own stages
pipeline.add_stage(PassThruStage(config))
pipeline.add_stage(RecipientFeaturesStage(config))

# Add monitor to record the performance of our new stage
pipeline.add_stage(MonitorStage(config))
```

## Predicting Fraudulent Emails with Accelerated Machine Learning.

For this example, we'll be using the `RecipientFeaturesStage` from the previous example in a real-world pipeline to detect fraudulent emails. The pipeline we will be building makes use of the `TritonInferenceStage` which is a pre-defined Morpheus stage designed to support the execution of Natural Language Processing (NLP) models via NVIDIA's [Triton inference server framework](https://developer.nvidia.com/nvidia-triton-inference-server), which allows for GPU accelerated ML/DL and seamless co-location and execution of a wide variety of model frameworks. For our application, we will be using the `phishing-bert-onnx` model, which is included with Morpheus in the `models/triton-model-repo/` directory.

The important thing to note is that the Triton server lives outside of the execution of the Morpheus pipeline and cannot be running on the same machine(s) that the pipeline is executing on. Instead, Triton is a service that we are communicating with via HTTP & [gRPC](https://grpc.io/) network protocols to interact with machine learning models hosted by Triton.

## Launching Triton

Triton will need to be running while we execute our pipeline. For simplicity, we will launch it locally inside of a Docker container.

Note: This step assumes you have both [Docker](https://docs.docker.com/engine/install/) and the [Nvidia container toolkit](https://docs.docker.com/engine/install/) installed.

From the root of the Morpheus project we will launch a Triton Docker container mounting the models directory in the container.

```shell
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --log-info=true
```

Once we have Triton running, we can verify that it is healthy using [curl](https://curl.se/); the `/v2/health/live` endpoint should return a 200 status code:

```shell
curl -v "localhost:8000/v2/health/live"
```

We can also query Triton for available models:

```shell
curl -X POST "localhost:8000/v2/repository/index"
```

Along  information about the `phishing-bert-onnx` model which we are going to be using:

```shell
curl "localhost:8000/v2/models/phishing-bert-onnx/config"
{"name":"phishing-bert-onnx","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"input_ids","datatype":"INT64","shape":[-1,128]},{"name":"attention_mask","datatype":"INT64","shape":[-1,128]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,2]}]}
```

Here is the expected length of the model inputs indicated by `"shape":[-1,128]}`.

## Defining our Pipeline
Set up our file paths for our input and output files. For simplicity, we assume that the `MORPHEUS_ROOT` environment variable is set to the root of the Morpheus project repository. In a production deployment, it may be more prudent to replace our usage of environment variables with command-line flags.

```python
root_dir = os.environ['MORPHEUS_ROOT']
out_dir = os.environ.get('OUT_DIR', '/tmp')

data_dir = os.path.join(root_dir, 'data')
labels_file = os.path.join(data_dir, 'labels_phishing.txt')
vocab_file = os.path.join(data_dir, 'bert-base-uncased-hash.txt')

input_file = os.path.join(root_dir, 'examples/data/email.jsonlines')
results_file = os.path.join(out_dir, 'detections.jsonlines')
```

To start, we will need to set a few parameters on the config object, these are parameters which are global to the pipeline as a whole, while individual stages receive their own parameters..

```python
config = Config()
config.mode = PipelineModes.NLP
config.num_threads = psutil.cpu_count()
config.feature_length = 128

with open(labels_file) as fh:
    config.class_labels = [x.strip() for x in fh]
```

We set our pipeline mode to NLP. Next, we use the third-party [psutils](https://psutil.readthedocs.io/en/stable/) library to set the `num_threads` property to match the number of cores in our system.

The `feature_length` property needs to match the length of the model inputs, which we queried from the model's config endpoint in the previous section.

Ground truth classification labels are read from the `data/labels_phishing.txt` file included in Morpheus.

Now that our config object is populated we move on to the pipeline itself. We are using the same input file from the previous examples, and to tokenize the input data we add Morpheus' `PreprocessNLPStage` with the `data/bert-base-uncased-hash.txt` vocabulary file.

```python
pipeline.add_stage(
    PreprocessNLPStage(
        config,
        vocab_hash_file=vocab_file,
        truncation=True,
        do_lower_case=True,
        add_special_tokens=False))
```

This stage uses the [cudf subword tokenizer](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.core.subword_tokenizer.SubwordTokenizer.__call__.html) to transform strings into a tensor of numbers to be fed into the neural network model. Rather than split the string by characters or whitespaces, we split them into meaningful subwords based upon the occurrence of the subwords in a large training corpus. You can find more details here: [https://arxiv.org/abs/1810.04805v2](https://arxiv.org/abs/1810.04805v2). Briefly, the text is converted to subword token ids based on the vocabulary file from a pretrained tokenizer. This stage contains the parameters to truncate the length of the text to a max number of tokens (`truncation=True`), change the casing to all lowercase (`do_lower_case=True`), and the choice to add the default BERT special tokens like `[SEP]` for separation between two sentences and `[CLS]` at the start of the text (not used here but enabled with: `add_special_tokens=True`). The tokenizer parameters and vocabulary hash file should exactly match what was used for tokenization during the training of the NLP model.

At this point, we have a pipeline that reads in a set of records and preprocesses them with the required metadata for our classifier to predict on. Our next step is to define a stage that applies a machine learning model to our `MessageMeta` object. To accomplish this, we will be using Morpheus' `TritonInferenceStage` class which will communicate with the `phishing-bert-onnx` model, which we already loaded into Triton.

Next, we will add a monitor stage to measure the inference rate, and a filter stage to filter out any results below a probability threshold of `0.9`.
```python
# Add a inference stage
pipeline.add_stage(
    TritonInferenceStage(
        config,
        model_name='phishing-bert-onnx',
        server_url=triton_url,
        force_convert_inputs=True,
    ))

pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))

# Filter values lower than 0.9
pipeline.add_stage(FilterDetectionsStage(config, threshold=0.9))
```

Lastly, we will save our results to disk. For this purpose, we are using two stages often used in conjunction with each other, the `SerializeStage` and the `WriteToFileStage`. `SerializeStage` is used to include and exclude columns desired in the output in addition to this converting from the `MultiMessage` class used by the previous stages to the `MessageMeta` class expected by the `WriteToFileStage`. The `WriteToFileStage` will append message data of the output file as messages are received; however, for performance, the `WriteToFileStage` does not flush contents out to disk for each message received, allowing the underlying [buffered output stream to flush as needed](https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html), and closing the file handle on shutdown.

```python
# Write the file to the output
pipeline.add_stage(SerializeStage(config))
pipeline.add_stage(WriteToFileStage(config, filename=results_file, overwrite=True))
```

Note that we didn't specify the output format. In our example, the result file contains the extension `.jsonlines`. Morpheus will infer the output format based on the extension. At time of writing the extensions that Morpheus will infer are: `.csv`, `.json` & `.jsonlines`

To explicitly set the output format we could specify the `file_type` argument to the `WriteToFileStage` which is an enumeration defined in `morpheus._lib.file_types.FileTypes`. Current values defined are:
* `FileTypes.Auto`
* `FileTypes.JSON`
* `FileTypes.CSV`

## The Completed Pipeline

```python
import logging
import os

import psutil

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import FilterDetectionsStage
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from morpheus.utils.logging import configure_logging

from recipient_feature_stage import RecipientFeaturesStage

def run_pipeline():
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    triton_url = os.environ.get('TRITON_URL', 'localhost:8001')
    root_dir = os.environ['MORPHEUS_ROOT']
    out_dir = os.environ.get('OUT_DIR', '/tmp')

    data_dir = os.path.join(root_dir, 'data')
    labels_file = os.path.join(data_dir, 'labels_phishing.txt')
    vocab_file = os.path.join(data_dir, 'bert-base-uncased-hash.txt')

    input_file = os.path.join(root_dir, 'examples/data/email.jsonlines')
    results_file = os.path.join(out_dir, 'detections.jsonlines')


    # Its necessary to configure the pipeline for NLP mode
    config = Config()
    config.mode = PipelineModes.NLP

    # Set the thread count to match our cpu count
    config.num_threads = psutil.cpu_count()
    config.feature_length = 128

    with open(labels_file) as fh:
        config.class_labels = [x.strip() for x in fh]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add our custom stage
    pipeline.add_stage(RecipientFeaturesStage(config))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Tokenize the input
    pipeline.add_stage(
        PreprocessNLPStage(
            config,
            vocab_hash_file=vocab_file,
            truncation=True,
            do_lower_case=True,
            add_special_tokens=False))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name='phishing-bert-onnx',
            server_url=triton_url,
            force_convert_inputs=True,
        ))

    # Monitor the inference rate
    pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))

    # Filter values lower than 0.9
    pipeline.add_stage(FilterDetectionsStage(config, threshold=0.9))

    # Write the to the output file
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=results_file, overwrite=True))

    # Run the pipeline
    pipeline.run()

if __name__ == "__main__":
    run_pipeline()
```

## Stage constructors

In our previous examples, we didn't define a constructor for the Python classes we were building for our stages. However, there are many cases where we will need to receive configuration parameters. Every stage constructor must receive an instance of a `morpheus.config.Config` object as its first argument and is then free to define additional stage-specific arguments after that. The Morpheus config object will contain configuration parameters needed by multiple stages in the pipeline, and the constructor in each Morpheus stage is free to inspect these. In contrast, parameters specific to a single stage are typically defined as constructor arguments.

It is also important to perform necessary validation checks in the constructor, allowing us to fail early rather than after the pipeline has started. In our `RecipientFeaturesStage` example, we hard-coded the Bert separator token; we could refactor the code a bit to receive that as a constructor argument, we could also use that opportunity to verify that the pipeline mode is set to `morpheus.config.PipelineModes.NLP`  our new refactored class definition now looks like:

```python
class RecipientFeaturesStage(SinglePortStage):
    def __init__(self,
                 config: Config,
                 sep_token: str='[SEP]'):
        super().__init__(config)
        if config.mode != PipelineModes.NLP:
            raise RuntimeError("RecipientFeaturesStage must be used in a pipeline configured for NLP")

        if len(sep_token):
            self._sep_token = sep_token
        else:
            raise ValueError("sep_token cannot be an empty string")
```

## Defining A New Source Stage

Creating a new source stage is similar to defining any other stage with a few differences. First, we will be subclassing `SingleOutputSource`. Second, the required methods are just the `name` property method and the `_build_source` method.

In this example, we will create a source that reads messages from a [RabbitMQ](https://www.rabbitmq.com/) queue using the [pika](https://pika.readthedocs.io/en/stable/#) client for Python. For simplicity, we will assume that authentication is not required for our RabbitMQ exchange. The body of the RabbitMQ messages will be JSON formatted; however, both authentication and support for other formats could be easily added later.

The `_build_source` method is similar to the `_build_single` method. It receives an instance of the pipeline segment; however, unlike in the previous examples, source stages don't have parent stages. As such, `_build_source` doesn't receive a `StreamPair`; however, it does return a `StreamPair`. In this case, we will define a node method `source_generator`, rather than calling `make_node` in our previous examples for a source stage; we will instead call `make_source`.

```python
def _build_source(self, seg: neo.Segment) -> StreamPair:
    node = seg.make_source(self.unique_name, self.source_generator)
    return node, MessageMeta
```

The `source_generator`  method is where most of the RabbitMQ specific code exists. Node methods receive an instance of `neo.Subscriber` as their first argument, when we have a message we wish to emit we call the `neo.Subscriber.on_next` method.

```python
def source_generator(self, subscriber: neo.Subscriber):
    try:
        while subscriber.is_subscribed():
            (method_frame, header_frame, body) = self._channel.basic_get(self._queue_name)
            if method_frame is not None:
                try:
                    buffer = StringIO(body.decode("utf-8"))
                    df = cudf.io.read_json(buffer, orient='records', lines=True)
                    subscriber.on_next(MessageMeta(df=df))
                except Exception as ex:
                    logger.exception("Error occurred converting RabbitMQ message to Dataframe: {}".format(ex))
                finally:
                    self._channel.basic_ack(method_frame.delivery_tag)
            else:
                # queue is empty, sleep before polling again
                time.sleep(self._poll_interval.total_seconds())

    finally:
        self._connection.close()
```

Note that when we read messages as quickly as we can from the queue; when the queue is empty, we call `time.sleep`, allowing for a context switch to occur if needed.

We only acknowledge the message by calling `basic_ack` once we have successfully emitted the message or failed to deserialize the message. If the pipeline shuts down after we have received a message but prior to us being able to call `on_next` . This means that we are avoiding the potential of a lost message at the risk of a potentially duplicate message if the pipeline is shut down after we have called `on_next` but before calling `basic_ack`.

## The Completed Source Stage

```python
import logging
import time
from datetime import timedelta
from io import StringIO

import neo
import pika

import cudf

from morpheus.config import Config
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)

class RabbitMQSourceStage(SingleOutputSource):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str
        RabbitMQ exchange type; defaults to `fanout`.
    queue_name : str
        Name of the queue to listen to. If left blank, RabbitMQ will generate a random queue name bound to the exchange.
    poll_interval : timedelta
        Amount of time  between polling RabbitMQ for new messages; defaults to 100ms
    """
    def __init__(self,
                 config: Config,
                 host: str,
                 exchange: str,
                 exchange_type: str='fanout',
                 queue_name: str='',
                 poll_interval: timedelta=None):
        super().__init__(config)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host))

        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=exchange, exchange_type=exchange_type)

        result = self._channel.queue_declare(
            queue=queue_name, exclusive=True)

        # When queue_name='' we will receive a randomly generated queue name
        self._queue_name = result.method.queue

        self._channel.queue_bind(
            exchange=exchange, queue=self._queue_name)

        if poll_interval is not None:
            self._poll_interval = poll_interval
        else:
            self._poll_interval = timedelta(milliseconds=100)

    @property
    def name(self) -> str:
        return "from-rabbitmq"

    def _build_source(self, seg: neo.Segment) -> StreamPair:
        node = seg.make_source(self.unique_name, self.source_generator)
        return node, MessageMeta

    def source_generator(self, subscriber: neo.Subscriber):
        try:
            while subscriber.is_subscribed():
                (method_frame, header_frame, body) = self._channel.basic_get(self._queue_name)
                if method_frame is not None:
                    try:
                        buffer = StringIO(body.decode("utf-8"))
                        df = cudf.io.read_json(buffer, orient='records', lines=True)
                        subscriber.on_next(MessageMeta(df=df))
                    except Exception as ex:
                        logger.exception("Error occurred converting RabbitMQ message to Dataframe: {}".format(ex))
                    finally:
                        self._channel.basic_ack(method_frame.delivery_tag)
                else:
                    # queue is empty, sleep before polling again
                    time.sleep(self._poll_interval.total_seconds())

        finally:
            self._connection.close()
```

## Defining A New Sink Stage

In Morpheus, we define sink stages as those stages output the results of a pipeline to a destination external to the pipeline. Morpheus includes two sink stages, a `WriteToFileStage` and a `WriteToKafkaStage`.

In the previous example, we wrote a `RabbitMQSourceStage`; we will complement that by writing a sink stage that can write Morpheus data into RabbitMQ. For this example, we are again using the [pika](https://pika.readthedocs.io/en/stable/#) client for Python.

The code for our sink will look similar to other stages with a few changes. First, we will subclass `SinglePortStage`:

```python
class WriteToRabbitMQStage(SinglePortStage):
```

In our `_build_single` we will be making use of the `make_sink` method rather than `make_node` or `make_source`
```python
def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
    node = seg.make_sink(self.unique_name, self.on_data, self.on_error, self.on_complete)
    seg.make_edge(input_stream[0], node)
    return input_stream
```

Note that in this case, while we created an edge from the parent node to our new node, we returned the `input_stream` unchanged. This allows for other sinks to be attached to our same parent node. We could hypothetically have a pipeline where we emit the results to both RabbitMQ and a file. In this situation, both would be children of the same upstream node.

![Morpheus node dependency diagram](img/sink_deps.png)

Similar to our previous examples, most of the actual business logic of the stage is contained in the `on_data` method. In this case, we grab a reference to the [cuDF](https://docs.rapids.ai/api/cudf/stable/) [DataFrame](https://docs.rapids.ai/api/cudf/stable/api_docs/dataframe.html) attached to the incoming message. We then serialize to an [io.StringIO](https://docs.python.org/3.8/library/io.html?highlight=stringio#io.StringIO) buffer, which is then sent to RabbitMQ.

```python
def on_data(self, message: MessageMeta):
    df = message.df
    buffer = StringIO()
    df.to_json(buffer, orient='records', lines=True)
    body = buffer.getvalue().strip()
    self._channel.basic_publish(exchange=self._exchange, routing_key=self._routing_key, body=body)
    return message
```

The two new methods introduced in this example are the `on_error` and `on_complete` methods. For both methods, we want to make sure that the [connection](https://pika.readthedocs.io/en/stable/modules/connection.html) object is properly closed.
Note: we didn't close the channel object since closing the connection will also close any associated channel objects.

```python
def on_error(self, ex: Exception):
    logger.exception("Error occurred : {}".format(ex))
    self._connection.close()

def on_complete(self):
    self._connection.close()
```

## The Completed Sink Stage

```python
import logging
import typing
from io import StringIO

import neo
import pika

import cudf

from morpheus.config import Config
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)

class WriteToRabbitMQStage(SinglePortStage):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str
        RabbitMQ exchange type; defaults to `fanout`.
    routing_key : str
        RabbitMQ routing key if needed.
    """
    def __init__(self,
                 config: Config,
                 host: str,
                 exchange: str,
                 exchange_type: str='fanout',
                 routing_key: str=''):
        super().__init__(config)
        self._connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host))

        self._exchange = exchange
        self._routing_key = routing_key

        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=self._exchange, exchange_type=exchange_type)


    @property
    def name(self) -> str:
        return "to-rabbitmq"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        node = seg.make_sink(self.unique_name, self.on_data, self.on_error, self.on_complete)
        seg.make_edge(input_stream[0], node)
        return input_stream

    def on_data(self, message: MessageMeta):
        df = message.df
        buffer = StringIO()
        df.to_json(buffer, orient='records', lines=True)
        body = buffer.getvalue().strip()
        self._channel.basic_publish(exchange=self._exchange, routing_key=self._routing_key, body=body)
        return message

    def on_error(self, ex: Exception):
        logger.exception("Error occurred : {}".format(ex))
        self._connection.close()

    def on_complete(self):
        self._connection.close()
```
