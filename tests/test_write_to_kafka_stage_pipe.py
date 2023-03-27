#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import typing

import numpy as np
import pytest

import cudf

from morpheus.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS

if (typing.TYPE_CHECKING):
    from kafka import KafkaConsumer


@pytest.mark.kafka
def test_write_to_kafka_stage_pipe(config,
                                   kafka_bootstrap_servers: str,
                                   kafka_consumer: "KafkaConsumer",
                                   kafka_topics: typing.Tuple[str, str]) -> None:
    """
    Even though WriteToKafkaStage only has a Python impl, testing with both C++ and Python execution
    to ensure it works just as well with the C++ impls of the message classes.
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(
        WriteToKafkaStage(config,
                          bootstrap_servers=kafka_bootstrap_servers,
                          output_topic=kafka_topics.output_topic,
                          client_id='morpheus_test_write_to_kafka_stage_pipe'))
    pipe.run()

    input_data = read_file_to_df(input_file, file_type=FileTypes.Auto).values

    kafka_messages = list(kafka_consumer)
    assert len(kafka_messages) == len(input_data)

    pdf = cudf.io.read_json("\n".join(rec.value.decode("utf-8") for rec in kafka_messages), lines=True).to_pandas()

    output_data = pdf.values
    assert len(output_data) == len(input_data)

    # Somehow 0.7 ends up being 0.7000000000000001
    input_data = np.around(input_data, 2)
    output_data = np.around(output_data, 2)

    assert output_data.tolist() == input_data.tolist()
