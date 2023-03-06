# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
import typing

import confluent_kafka as ck
import mrc
import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_control import MessageControl
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.kafka_source_stage import AutoOffsetReset

logger = logging.getLogger(__name__)


@register_stage("from-cm-kafka", modes=[PipelineModes.AE])
class ControlMessageKafkaSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Load control messages from a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bootstrap_servers : str
        Comma-separated list of bootstrap servers. If using Kafka created via `docker-compose`, this can be set to
        'auto' to automatically determine the cluster IPs and ports
    input_topic : str
        Input kafka topic.
    group_id : str
        Specifies the name of the consumer group a Kafka consumer belongs to.
    client_id : str, default = None
        An optional identifier of the consumer.
    poll_interval : str
        Seconds that elapse between polling Kafka for new messages. Follows the pandas interval format.
    disable_commit : bool, default = False
        Enabling this option will skip committing messages as they are pulled off the server. This is only useful for
        debugging, allowing the user to process the same messages multiple times.
    disable_pre_filtering : bool, default = False
        Enabling this option will skip pre-filtering of json messages. This is only useful when inputs are known to be
        valid json.
    auto_offset_reset : `AutoOffsetReset`, case_sensitive = False
        Sets the value for the configuration option 'auto.offset.reset'. See the kafka documentation for more
        information on the effects of each value."
    stop_after: int, default = 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    async_commits: bool, default = True
        Enable commits to be performed asynchronously. Ignored if `disable_commit` is `True`.
    """

    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_cm",
                 group_id: str = "morpheus",
                 client_id: str = None,
                 poll_interval: str = "10millis",
                 disable_commit: bool = False,
                 disable_pre_filtering: bool = False,
                 auto_offset_reset: AutoOffsetReset = AutoOffsetReset.LATEST,
                 stop_after: int = 0,
                 async_commits: bool = True):

        super().__init__(c)

        if isinstance(auto_offset_reset, AutoOffsetReset):
            auto_offset_reset = auto_offset_reset.value

        self._consumer_params = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'session.timeout.ms': "60000",
            "auto.offset.reset": auto_offset_reset
        }
        if client_id is not None:
            self._consumer_params['client.id'] = client_id

        self._topic = input_topic
        # Setting max batch size to 1. As this source recieves only task defination (control messages)
        self._max_batch_size = 1
        self._max_concurrent = c.num_threads
        self._disable_commit = disable_commit
        self._disable_pre_filtering = disable_pre_filtering
        self._stop_after = stop_after
        self._async_commits = async_commits
        self._client = None

        # Flag to indicate whether or not we should stop
        self._stop_requested = False

        self._poll_interval = pd.Timedelta(poll_interval).total_seconds()
        self._started = False

        self._records_emitted = 0
        self._num_messages = 0

    @property
    def name(self) -> str:
        return "from-cm-kafka"

    def supports_cpp_node(self):
        return False

    def _process_msg(self, consumer, msg):

        control_messages = []

        payload = msg.value()
        if payload is not None:

            try:
                decoded_msg = payload.decode("utf-8")
                control_messages_conf = json.loads(decoded_msg)
                self._num_messages += 1
                for control_message_conf in control_messages_conf.get("inputs", []):
                    self._records_emitted += 1
                    control_messages.append(MessageControl(control_message_conf))
            except Exception as e:
                logger.error("\nError converting payload to MessageControl : {}".format(e))

        if (not self._disable_commit):
            consumer.commit(message=msg, asynchronous=self._async_commits)

        if self._stop_after > 0 and self._records_emitted >= self._stop_after:
            self._stop_requested = True

        return control_messages

    def _source_generator(self):
        consumer = None
        try:
            consumer = ck.Consumer(self._consumer_params)
            consumer.subscribe([self._topic])

            while not self._stop_requested:

                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    do_sleep = True

                else:
                    msg_error = msg.error()
                    if msg_error is None:
                        control_messages = self._process_msg(consumer, msg)
                        for control_message in control_messages:
                            yield control_message

                    elif msg_error == ck.KafkaError._PARTITION_EOF:
                        do_sleep = True
                    else:
                        raise ck.KafkaException(msg_error)

                if do_sleep and not self._stop_requested:
                    time.sleep(self._poll_interval)

        finally:
            # Close the consumer and call on_completed
            if (consumer):
                consumer.close()

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        source = builder.make_source(self.unique_name, self._source_generator)

        return source, MessageControl
