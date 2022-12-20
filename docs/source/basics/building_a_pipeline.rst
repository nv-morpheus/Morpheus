..
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

Building a Pipeline
===================

To build a pipeline via the CLI, users must first specify the type of pipeline, a source object, followed by a sequential list of stages. For each stage, options can be specified to configure the particular stage. Since stages are listed sequentially the output of one stage becomes the input to the next. Unless heavily customized, pipelines will start with either:

.. code-block:: bash

   # For NLP Pipelines
   morpheus run pipeline-nlp ...
   # For FIL Pipelines
   morpheus run pipeline-fil ...

While each stage will have configuration options, there are options that apply to the pipeline as a whole as well. Check
``morpheus run --help``, ``morpheus run pipeline-nlp --help`` and ``morpheus run pipeline-fil --help`` for these global
Pipeline options.

Source Stages
^^^^^^^^^^^^^

All pipelines configured with the CLI need to start with a source object. Currently Morpheus supports two source types:

 * ``from-kafka``
    - Pulls messages from a Kafka cluster into the Pipeline
    - Kafka cluster can be remote or local
    - Refer to :py:obj:`~morpheus.pipeline.input.from_kafka.KafkaSourceStage` for more information
 * ``from-file``
    - Reads from a local file into the Pipeline
    - Supports JSON lines format
    - All lines are read at the start and queued into the pipeline at one time. Useful for performance testing.
    - Refer to :py:obj:`~morpheus.pipeline.input.from_file.FileSourceStage` for more information

Stages
^^^^^^

From this point on, any number of stages can be sequentially added to the command line from start to finish. For example, to build a simple pipeline that reads from kafka, deserializes messages, serializes them, and then writes to a file, use the following:

.. code-block:: console

   $ morpheus --log_level=DEBUG run pipeline-nlp \
      from-kafka --input_topic test_pcap \
      deserialize \
      serialize \
      to-file --filename .tmp/temp_out.json
   ...
   ====Building Pipeline====
   Added source: from-kafka-0
     └─> cudf.DataFrame
   Added stage: deserialize-1
     └─ cudf.DataFrame -> morpheus.MultiMessage
   Added stage: serialize-2
     └─ morpheus.MultiMessage -> List[str]
   Added stage: to-file-3
     └─ List[str] -> List[str]
   ====Building Pipeline Complete!====
   ...

After the ``====Building Pipeline====`` message, if logging is ``INFO`` or greater, the CLI will print a list of all
stages and the type transformations of each stage. To be a valid Pipeline, the output type of one stage must match the
input type of the next. Many stages are flexible and will determine their type at runtime but some stages require a
specific input type. If your Pipeline is configured incorrectly, Morpheus will report the error. For example, if we run
the same command as above but forget the ``serialize`` stage, the following will be displayed:

.. code-block:: console

   $ morpheus --log_level=DEBUG run pipeline-nlp \
      from-kafka --input_topic test_pcap \
      deserialize \
      to-file --filename .tmp/temp_out.json
   ...

   ====Building Pipeline====
   Added source: from-file-0
     └─> cudf.DataFrame
   Added stage: buffer-1
     └─ cudf.DataFrame -> cudf.DataFrame
   Error occurred during Pipeline.build(). Exiting.
   RuntimeError: The preprocess-nlp stage cannot handle input of <class 'cudf.core.dataframe.DataFrame'>. Accepted input types: (<class 'morpheus.pipeline.messages.MultiMessage'>, typing.StreamFuture[morpheus.pipeline.messages.MultiMessage])

This indicates that the ``to-file`` stage cannot accept the input type of `morpheus.pipeline.messages.MultiMessage`.
This is because the ``to-file`` stage has no idea how to write that class to a file, it only knows how to write strings.
To ensure you have a valid pipeline, examine the ``Accepted input types: (typing.List[str],)`` portion of the message.
This indicates you need a stage that converts from the output type of the ``deserialize`` stage,
`morpheus.pipeline.messages.MultiMessage`, to `typing.List[str]`, which is exactly what the ``serialize`` stage does.

Available Stages
^^^^^^^^^^^^^^^^

For a complete list of available stages, use the CLI help commands. The available stages can also be queried from the CLI using ``morpheus run pipeline-nlp --help`` or ``morpheus run pipeline-fil --help``.
