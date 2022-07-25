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

from subprocess import Popen
from typing import Tuple

from kafka import KafkaConsumer
from kafka import KafkaProducer


def write_to_kafka(kafka_server: Tuple[Popen, int], message: bytes) -> None:
    """Write a message to kafka_server."""
    _, kafka_port = kafka_server
    producer = KafkaProducer(bootstrap_servers='localhost:{}'.format(kafka_port))
    producer.send('morpheus-test', message)
    producer.flush()


def test_write_and_read(kafka_server: Tuple[Popen, int], kafka_consumer: KafkaConsumer) -> None:
    """Write to kafka_server, consume with kafka_consumer."""
    message = b'msg'
    write_to_kafka(kafka_server, message)
    consumed = list(kafka_consumer)
    assert len(consumed) == 1
    assert consumed[0].topic == 'morpheus-test'
    assert consumed[0].value == message
