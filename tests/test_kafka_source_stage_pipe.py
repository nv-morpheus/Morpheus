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

import mrc
import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.dfp_length_checker import DFPLengthChecker
from utils import TEST_DIRS
from utils import assert_results
from utils import write_data_to_kafka
from utils import write_file_to_kafka


@pytest.mark.kafka
def test_kafka_source_stage_pipe(config, kafka_bootstrap_servers: str, kafka_topics: typing.Tuple[str, str]) -> None:
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines")

    # Fill our topic with the input data
    num_records = write_file_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, input_file)

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         client_id='morpheus_kafka_source_stage_pipe',
                         stop_after=num_records))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, input_file))
    pipe.run()

    assert_results(comp_stage.get_results())


class OffsetChecker(SinglePortStage):
    """
    Verifies that the kafka offsets are being updated as a way of verifying that the
    consumer is performing a commit.
    """

    def __init__(self, c: Config, bootstrap_servers: str, group_id: str):
        super().__init__(c)

        # importing here so that running without the --run_kafka flag won't fail due
        # to not having the kafka libs installed
        from kafka import KafkaAdminClient

        self._client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        self._group_id = group_id
        self._offsets = None

    @property
    def name(self) -> str:
        return "morpheus_offset_checker"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _offset_checker(self, x):
        at_least_one_gt = False
        new_offsets = self._client.list_consumer_group_offsets(self._group_id)

        if self._offsets is not None:
            for (tp, prev_offset) in self._offsets.items():
                new_offset = new_offsets[tp]

                assert new_offset.offset >= prev_offset.offset

                if new_offset.offset > prev_offset.offset:
                    at_least_one_gt = True

            assert at_least_one_gt

        self._offsets = new_offsets

        return x

    def _build_single(self, builder: mrc.Builder, input_stream):
        node = builder.make_node(self.unique_name, self._offset_checker)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]


@pytest.mark.kafka
@pytest.mark.parametrize('num_records', [10, 100, 1000])
def test_kafka_source_commit(num_records, config, kafka_bootstrap_servers: str,
                             kafka_topics: typing.Tuple[str, str]) -> None:

    data = [{'v': i} for i in range(num_records)]
    num_written = write_data_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, data)
    assert num_written == num_records

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         group_id='morpheus',
                         client_id='morpheus_kafka_source_commit',
                         stop_after=num_records,
                         async_commits=False))

    pipe.add_stage(OffsetChecker(config, bootstrap_servers=kafka_bootstrap_servers, group_id='morpheus'))
    pipe.add_stage(TriggerStage(config))

    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, pd.DataFrame(data=data), include=[r'^v$'], reset_index=True))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.kafka
@pytest.mark.parametrize('num_records', [1000])
def test_kafka_source_batch_pipe(config,
                                 kafka_bootstrap_servers: str,
                                 kafka_topics: typing.Tuple[str, str],
                                 num_records: int) -> None:
    data = [{'v': i} for i in range(num_records)]
    num_written = write_data_to_kafka(kafka_bootstrap_servers, kafka_topics.input_topic, data)
    assert num_written == num_records

    expected_length = config.pipeline_batch_size
    num_exact = num_records // expected_length

    pipe = LinearPipeline(config)
    pipe.set_source(
        KafkaSourceStage(config,
                         bootstrap_servers=kafka_bootstrap_servers,
                         input_topic=kafka_topics.input_topic,
                         auto_offset_reset="earliest",
                         poll_interval="1seconds",
                         client_id='morpheus_kafka_source_stage_pipe',
                         stop_after=num_records))
    pipe.add_stage(DFPLengthChecker(config, expected_length=expected_length, num_exact=num_exact))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, pd.DataFrame(data=data), include=[r'^v$'], reset_index=True))
    pipe.run()

    assert_results(comp_stage.get_results())
