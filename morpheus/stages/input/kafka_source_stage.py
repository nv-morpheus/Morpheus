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

import neo
import pandas as pd
from cudf_kafka._lib.kafka import KafkaDatasource

import cudf

import morpheus._lib.stages as neos
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
    """

    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "custreamz",
                 poll_interval: str = "10millis",
                 disable_commit: bool = False,
                 disable_pre_filtering: bool = False):
        super().__init__(c)

        self._consumer_conf = {
            'bootstrap.servers': bootstrap_servers, 'group.id': group_id, 'session.timeout.ms': "60000"
        }

        self._input_topic = input_topic
        self._poll_interval = poll_interval
        self._max_batch_size = c.pipeline_batch_size
        self._max_concurrent = c.num_threads
        self._disable_commit = disable_commit
        self._disable_pre_filtering = disable_pre_filtering
        self._client = None

        # What gets passed to streamz kafka
        topic = self._input_topic
        consumer_params = self._consumer_conf
        poll_interval = self._poll_interval
        npartitions = None
        refresh_partitions = False
        max_batch_size = self._max_batch_size
        keys = False
        engine = None

        self._consumer_params = consumer_params
        # Override the auto-commit config to enforce custom streamz checkpointing
        self._consumer_params['enable.auto.commit'] = 'false'
        if 'auto.offset.reset' not in self._consumer_params.keys():
            self._consumer_params['auto.offset.reset'] = 'latest'
        self._topic = topic
        self._npartitions = npartitions
        self._refresh_partitions = refresh_partitions
        if self._npartitions is not None and self._npartitions <= 0:
            raise ValueError("Number of Kafka topic partitions must be > 0.")
        self._poll_interval = pd.Timedelta(poll_interval).total_seconds()
        self._max_batch_size = max_batch_size
        self._keys = keys
        self._engine = engine
        self._started = False

    @property
    def name(self) -> str:
        return "from-kafka"

    def supports_cpp_node(self):
        return True

    def _source_generator(self, s: neo.Subscriber):
        # Each invocation of this function makes a new thread so recreate the producers

        # Set some initial values
        npartitions = self._npartitions
        consumer = None
        consumer_params = self._consumer_params

        try:

            # Now begin the script
            import confluent_kafka as ck

            if self._engine == "cudf":  # pragma: no cover
                from custreamz import kafka

            if self._engine == "cudf":  # pragma: no cover
                consumer = kafka.Consumer(consumer_params)
            else:
                consumer = ck.Consumer(consumer_params)

            # weakref.finalize(self, lambda c=consumer: _close_consumer(c))
            tp = ck.TopicPartition(self._topic, 0, 0)

            attempts = 0
            max_attempts = 5

            # Attempt to connect to the cluster. Try 5 times before giving up
            while attempts < max_attempts:
                try:
                    # blocks for consumer thread to come up
                    consumer.get_watermark_offsets(tp)

                    logger.debug("Connected to Kafka source at '%s' on attempt #%d/%d",
                                 self._consumer_conf["bootstrap.servers"],
                                 attempts + 1,
                                 max_attempts)

                    break
                except (RuntimeError, ck.KafkaException):
                    attempts += 1

                    # Raise the error if we hit the max
                    if (attempts >= max_attempts):
                        logger.exception(("Error while getting Kafka watermark offsets. Max attempts (%d) reached. "
                                          "Check the bootstrap_servers parameter ('%s')"),
                                         max_attempts,
                                         self._consumer_conf["bootstrap.servers"])
                        raise
                    else:
                        logger.warning("Error while getting Kafka watermark offsets. Attempt #%d/%d",
                                       attempts,
                                       max_attempts,
                                       exc_info=True)

                        # Exponential backoff
                        time.sleep(2.0**attempts)

            try:
                if npartitions is None:

                    kafka_cluster_metadata = consumer.list_topics(self._topic)

                    if self._engine == "cudf":  # pragma: no cover
                        npartitions = len(kafka_cluster_metadata[self._topic.encode('utf-8')])
                    else:
                        npartitions = len(kafka_cluster_metadata.topics[self._topic].partitions)

                positions = [0] * npartitions

                tps = []
                for partition in range(npartitions):
                    tps.append(ck.TopicPartition(self._topic, partition))

                while s.is_subscribed():
                    try:
                        committed = consumer.committed(tps, timeout=1)
                    except ck.KafkaException:
                        pass
                    else:
                        for tp in committed:
                            positions[tp.partition] = tp.offset
                        break

                while s.is_subscribed():
                    out = []

                    if self._refresh_partitions:
                        kafka_cluster_metadata = consumer.list_topics(self._topic)

                        if self._engine == "cudf":  # pragma: no cover
                            new_partitions = len(kafka_cluster_metadata[self._topic.encode('utf-8')])
                        else:
                            new_partitions = len(kafka_cluster_metadata.topics[self._topic].partitions)

                        if new_partitions > npartitions:
                            positions.extend([-1001] * (new_partitions - npartitions))
                            npartitions = new_partitions

                    for partition in range(npartitions):

                        tp = ck.TopicPartition(self._topic, partition, 0)

                        try:
                            low, high = consumer.get_watermark_offsets(tp, timeout=0.1)
                        except (RuntimeError, ck.KafkaException):
                            continue

                        if 'auto.offset.reset' in consumer_params.keys():
                            if consumer_params['auto.offset.reset'] == 'latest' and positions[partition] == -1001:
                                positions[partition] = high

                        current_position = positions[partition]

                        lowest = max(current_position, low)

                        if high > lowest + self._max_batch_size:
                            high = lowest + self._max_batch_size
                        if high > lowest:
                            out.append((consumer_params, self._topic, partition, self._keys, lowest, high - 1))
                            positions[partition] = high

                    consumer_params['auto.offset.reset'] = 'earliest'

                    if (out):
                        for part in out:

                            meta = self._kafka_params_to_messagemeta(part)

                            # Once the meta goes out of scope, commit it
                            def commit(topic, part_no, keys, lowest, offset):
                                # topic, part_no, _, _, offset = part[1:]
                                try:
                                    _tp = ck.TopicPartition(topic, part_no, offset + 1)
                                    consumer.commit(offsets=[_tp], asynchronous=True)
                                except Exception:
                                    logger.exception(("Error occurred in `from-kafka` stage with "
                                                      "broker '%s' while committing message: %d"),
                                                     self._consumer_conf["bootstrap.servers"],
                                                     offset)

                            weakref.finalize(meta, commit, *part[1:])

                            # Push the message meta
                            s.on_next(meta)
                    else:
                        time.sleep(self._poll_interval)
            except Exception:
                logger.exception(("Error occurred in `from-kafka` stage with broker '%s' while processing messages"),
                                 self._consumer_conf["bootstrap.servers"])
                raise

        finally:
            # Close the consumer and call on_completed
            if (consumer):
                consumer.close()
            s.on_completed()

    def _kafka_params_to_messagemeta(self, x: tuple):

        # Unpack
        kafka_params, topic, partition, keys, low, high = x

        gdf = self._read_gdf(kafka_params, topic=topic, partition=partition, lines=True, start=low, end=high + 1)

        return MessageMeta(gdf)

    @staticmethod
    def _read_gdf(kafka_configs,
                  topic=None,
                  partition=0,
                  lines=True,
                  start=0,
                  end=0,
                  batch_timeout=10000,
                  delimiter="\n",
                  message_format="json"):
        """
        Replicates `custreamz.Consumer.read_gdf` function which does not work for some reason.
        """

        if topic is None:
            raise ValueError("ERROR: You must specifiy the topic "
                             "that you want to consume from")

        kafka_confs = {str.encode(key): str.encode(value) for key, value in kafka_configs.items()}

        kafka_datasource = None

        try:
            kafka_datasource = KafkaDatasource(
                kafka_confs,
                topic.encode(),
                partition,
                start,
                end,
                batch_timeout,
                delimiter.encode(),
            )

            cudf_readers = {
                "json": cudf.io.read_json,
                "csv": cudf.io.read_csv,
                "orc": cudf.io.read_orc,
                "avro": cudf.io.read_avro,
                "parquet": cudf.io.read_parquet,
            }

            result = cudf_readers[message_format](kafka_datasource, engine="cudf", lines=lines)

            return cudf.DataFrame(data=result._data, index=result._index)
        except Exception:
            logger.exception("Error occurred converting KafkaDatasource to Dataframe.")
        finally:
            if (kafka_datasource is not None):
                # Close up the cudf datasource instance
                # TODO: Ideally the C++ destructor should handle the
                # unsubscribe and closing the socket connection.
                kafka_datasource.unsubscribe()
                kafka_datasource.close(batch_timeout)

    def _build_source(self, seg: neo.Segment) -> StreamPair:

        if (self._build_cpp_node()):
            source = neos.KafkaSourceStage(seg,
                                           self.unique_name,
                                           self._max_batch_size,
                                           self._topic,
                                           int(self._poll_interval * 1000),
                                           self._consumer_params,
                                           self._disable_commit,
                                           self._disable_pre_filtering)
            source.concurrency = self._max_concurrent
        else:
            source = seg.make_source(self.unique_name, self._source_generator)

        source.concurrency = self._max_concurrent

        return source, MessageMeta
