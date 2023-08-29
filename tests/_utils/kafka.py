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

import json
import os
import subprocess
import time
import typing
from collections import namedtuple
from functools import partial

import pytest

# actual topic names not important, but we will need two of them.
KafkaTopics = namedtuple('KafkaTopics', ['input_topic', 'output_topic'])


@pytest.fixture(name='kafka_topics', scope='function')
def kafka_topics_fixture():
    yield KafkaTopics(f'morpheus_unittest_input_{time.time()}', f'morpheus_unittest_output_{time.time()}')


@pytest.fixture(scope="function")
def kafka_bootstrap_servers(kafka_server: typing.Tuple[subprocess.Popen, int]):
    """
    Used by tests that require both an input and an output topic
    """
    kafka_port = kafka_server[1]
    yield f"localhost:{kafka_port}"


def _wait_until(cond: typing.Callable[[], bool], timeout: float = 15, interval: float = 0.1):
    """Poll until the condition is True."""
    start = time.time()
    end = start + timeout
    while time.time() <= end:
        if cond() is True:
            return
        time.sleep(interval)

    raise AssertionError(f"Condition not true in {timeout} seconds")


@pytest.fixture(scope='function')
def kafka_consumer(kafka_bootstrap_servers: str, kafka_topics: KafkaTopics):
    import pytest_kafka
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(kafka_topics.output_topic,
                             bootstrap_servers=kafka_bootstrap_servers,
                             consumer_timeout_ms=500,
                             group_id=f'morpheus_unittest_reader_{time.time()}',
                             client_id=f'morpheus_unittest_reader_{time.time()}')

    def partitions_assigned():
        consumer.poll(timeout_ms=20)
        return len(consumer.assignment()) > 0

    _wait_until(partitions_assigned)

    consumer.seek_to_beginning()

    yield consumer


def init_pytest_kafka() -> dict:
    """
    Since the Kafka tests don't run by default, we will silently fail to initialize unless --run_kafka is enabled.
    """
    try:
        import pytest_kafka
        os.environ['KAFKA_OPTS'] = "-Djava.net.preferIPv4Stack=True"
        # Initialize pytest_kafka fixtures following the recomendations in:
        # https://gitlab.com/karolinepauls/pytest-kafka/-/blob/master/README.rst
        kafka_scripts = os.path.join(os.path.dirname(pytest_kafka.__file__), 'kafka/bin/')
        if not os.path.exists(kafka_scripts):
            # check the old location
            kafka_scripts = os.path.join(os.path.dirname(os.path.dirname(pytest_kafka.__file__)), 'kafka/bin/')

        kafka_bin = os.path.join(kafka_scripts, 'kafka-server-start.sh')
        zookeeper_bin = os.path.join(kafka_scripts, 'zookeeper-server-start.sh')

        for kafka_script in (kafka_bin, zookeeper_bin):
            if not os.path.exists(kafka_script):
                raise RuntimeError(f"Required Kafka script not found: {kafka_script}")

        teardown_fn = partial(pytest_kafka.terminate, signal_fn=subprocess.Popen.kill)
        zookeeper_proc = pytest_kafka.make_zookeeper_process(zookeeper_bin, teardown_fn=teardown_fn, scope='session')
        kafka_server = pytest_kafka.make_kafka_server(kafka_bin,
                                                      'zookeeper_proc',
                                                      teardown_fn=teardown_fn,
                                                      scope='session')

        return {
            "avail": True,
            "setup_error": "",
            "zookeeper_proc": zookeeper_proc,
            "kafka_server": kafka_server,
            "kafka_consumer": kafka_consumer,
            "kafka_topics": kafka_topics_fixture,
            "kafka_bootstrap_servers": kafka_bootstrap_servers
        }
    except Exception as e:
        return {"avail": False, "setup_error": e}


def write_data_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        data: typing.List[typing.Union[str, dict]],
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes `data` into a given Kafka topic, emitting one message for each line int he file. Returning the number of
    messages written
    """
    # pylint: disable=import-error
    from kafka import KafkaProducer
    num_records = 0
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers, client_id=client_id)
    for row in data:
        if isinstance(row, dict):
            row = json.dumps(row)
        producer.send(kafka_topic, row.encode('utf-8'))
        num_records += 1

    producer.flush()

    assert num_records > 0

    return num_records


def write_file_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        input_file: str,
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes data from `inpute_file` into a given Kafka topic, emitting one message for each line int he file.
    Returning the number of messages written
    """
    with open(input_file, encoding='UTF-8') as fh:
        data = [line.strip() for line in fh]

    return write_data_to_kafka(bootstrap_servers=bootstrap_servers,
                               kafka_topic=kafka_topic,
                               data=data,
                               client_id=client_id)
