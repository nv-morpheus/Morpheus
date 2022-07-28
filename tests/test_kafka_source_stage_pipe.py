#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from subprocess import Popen

import numpy as np
import pytest

from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import write_file_to_kafka


@pytest.mark.kafka
def test_kafka_source_stage_pipe(tmp_path,
                                 config,
                                 kafka_server: typing.Tuple[Popen, int],
                                 kafka_topics: typing.Tuple[str, str]) -> None:
    _, kafka_port = kafka_server
    bootstrap_servers = "localhost:{}".format(kafka_port)

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")
    out_file = os.path.join(tmp_path, 'results.jsonlines')

    # Fill our topic with the input data
    num_records = write_file_to_kafka(bootstrap_servers, kafka_topics.input_topic, input_file)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         stop_after=num_records))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert os.path.exists(out_file)

    input_data = read_file_to_df(input_file, file_type=FileTypes.Auto).values
    output_data = read_file_to_df(out_file, file_type=FileTypes.Auto).values

    assert len(input_data) == num_records
    assert len(output_data) == num_records

    # Somehow 0.7 ends up being 0.7000000000000001
    input_data = np.around(input_data, 2)
    output_data = np.around(output_data, 2)

    assert output_data.tolist() == input_data.tolist()
