#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class KafkaOffsetChecker(PassThruTypeMixin, SinglePortStage):
    """
    Verifies that the kafka offsets are being updated as a way of verifying that the
    consumer is performing a commit.
    """

    def __init__(self, c: Config, bootstrap_servers: str, group_id: str):
        super().__init__(c)

        # Importing here so that running without the --run_kafka flag won't fail due
        # to not having the kafka libs installed
        from kafka import KafkaAdminClient

        self._client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        self._group_id = group_id
        self._offsets = None

    @property
    def name(self) -> str:
        return "kafka_offset_checker"

    def accepted_types(self) -> (typing.Any, ):
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

    def _offset_checker(self, x: typing.Any) -> typing.Any:
        at_least_one_gt = False
        new_offsets = self._client.list_consumer_group_offsets(self._group_id)

        if self._offsets is not None:
            for (topic_partition, prev_offset) in self._offsets.items():
                new_offset = new_offsets[topic_partition]

                assert new_offset.offset >= prev_offset.offset

                if new_offset.offset > prev_offset.offset:
                    at_least_one_gt = True

            assert at_least_one_gt

        self._offsets = new_offsets

        return x

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._offset_checker))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
