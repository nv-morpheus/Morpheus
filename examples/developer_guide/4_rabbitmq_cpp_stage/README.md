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

# Example RabbitMQ stages
This example builds upon the `examples/developer_guide/2_2_rabbitmq` example adding a C++ implementation for the `RabbitMQSourceStage`.

This example adds two flags to the `read_simple.py` script. A `--use_cpp` flag which defaults to `True` and a `--num_threads` flag which defaults to the number of cores on the system as returned by `os.cpu_count()`.

## Testing with a RabbitMQ container
Testing can be performed locally with the RabbitMQ supplied docker image from the [RabbitMQ container registry](https://registry.hub.docker.com/_/rabbitmq/):
```bash
docker run --rm -it --hostname my-rabbit -p 15672:15672 -p 5672:5672 rabbitmq:3-management
```

The image can be verified with the web management console by opening http://localhost:15672 in a web browser. Enter "guest" for both the username and the password.

## Launch the reader
In a second terminal from the root of the morpheus repo execute:
```bash
python examples/developer_guide/4_rabbitmq_cpp_stage/read_simple.py
```

This will read from a RabbitMQ exchange named 'logs', and write the results to `/tmp/results.json`.

If no exchange named 'logs' exists in RabbitMQ it will be created.

## Launch the writer
In a third terminal from the root of the morpheus repo execute:
```bash
python examples/developer_guide/4_rabbitmq_cpp_stage/write_simple.py
```

This will read json data from the `examples/data/email.jsonlines` file and publish the data into the 'logs' RabbitMQ exchange as a single message.

The `write_simple.py` script will exit as soon as the message is written to the queue. The `read_simple.py` script on the otherhand will continue reading from the queue until explicitly shut down with a control-C.
