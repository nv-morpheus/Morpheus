# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import logging
import time
import weakref
from email import message

import confluent_kafka as ck
import pandas as pd
import srf

import cudf

import morpheus._lib.stages as _stages
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class KafkaSourceStage(SingleOutputSource):
    """
    Load messages from a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bootstrap_servers : str
        Kafka cluster bootstrap servers separated by a comma.
    input_topic : str
        Input kafka topic.
    group_id : str
        Specifies the name of the consumer group a Kafka consumer belongs to.
    poll_interval : str
        Seconds that elapse between polling Kafka for new messages. Follows the pandas interval format.
    disable_commit: bool, default = False
        Enabling this option will skip committing messages as they are pulled off the server. This is only useful for
        debugging, allowing the user to process the same messages multiple times.
    disable_pre_filtering : bool, default = False
        Enabling this option will skip pre-filtering of json messages. This is only useful when inputs are known to be
        valid json.
    auto_offset_reset : str, default = "latest"
        Sets the value for the configuration option 'auto.offset.reset'. See the kafka documentation for more
        information on the effects of each value."
    """

    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "morpheus",
                 poll_interval: str = "10millis",
                 disable_commit: bool = False,
                 disable_pre_filtering: bool = False,
                 auto_offset_reset: str = "latest"):
        super().__init__(c)

        self._consumer_params = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'session.timeout.ms': "60000",
            "auto.offset.reset": auto_offset_reset,
            'enable.auto.commit': 'false'
        }

        self._topic = input_topic
        self._max_batch_size = c.pipeline_batch_size
        self._max_concurrent = c.num_threads
        self._disable_commit = disable_commit
        self._disable_pre_filtering = disable_pre_filtering
        self._client = None

        # Flag to indicate whether or not we should stop
        self._stop_requested = False

        self._poll_interval = pd.Timedelta(poll_interval).total_seconds()
        self._started = False

    @property
    def name(self) -> str:
        return "from-kafka"

    def supports_cpp_node(self):
        return True

    def stop(self):

        # Indicate we need to stop
        self._stop_requested = True

        return super().stop()

    def _source_generator(self):
        # Each invocation of this function makes a new thread so recreate the producers

        # Set some initial values
        consumer = None
        try:
            consumer = ck.Consumer(self._consumer_params)
            consumer.subscribe([self._topic])

            while not self._stop_requested:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    time.sleep(self._poll_interval)
                    continue

                msg_error = msg.error()
                if msg_error is None:
                    payload = msg.value()
                    if payload is not None:
                        df = None
                        try:
                            df = cudf.io.read_json(payload, lines=True)
                        except Exception as e:
                            logger.error("Error parsing payload into a dataframe : {}".format(e))
                        finally:
                            if (not self._disable_commit):
                                consumer.commit(message=msg)

                        if df is not None:
                            yield MessageMeta(df)

                elif msg_error == ck.KafkaError._PARTITION_EOF:
                    time.sleep(self._poll_interval)
                else:
                    raise ck.KafkaException(msg_error)

        finally:
            # Close the consumer and call on_completed
            if (consumer):
                consumer.close()

    def _build_source(self, builder: srf.Builder) -> StreamPair:

        if (self._build_cpp_node()):
            source = _stages.KafkaSourceStage(builder,
                                              self.unique_name,
                                              self._max_batch_size,
                                              self._topic,
                                              int(self._poll_interval * 1000),
                                              self._consumer_params,
                                              self._disable_commit,
                                              self._disable_pre_filtering)

            # Only use multiple progress engines with C++. The python implementation will duplicate messages with
            # multiple threads
            source.launch_options.pe_count = self._max_concurrent
        else:
            source = builder.make_source(self.unique_name, self._source_generator())

        return source, MessageMeta
