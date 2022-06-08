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
import typing

import confluent_kafka as ck
import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class WriteToKafkaStage(SinglePortStage):
    """
    Write messages to a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    bootstrap_servers : str
        Kafka cluster bootstrap servers separated by comma.
    output_topic : str
        Output kafka topic.

    """

    def __init__(self, c: Config, bootstrap_servers: str, output_topic: str):
        super().__init__(c)

        self._kafka_conf = {'bootstrap.servers': bootstrap_servers}

        self._output_topic = output_topic
        self._poll_time = 0.2
        self._max_concurrent = c.num_threads

    @property
    def name(self) -> str:
        return "to-kafka"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def _build_single(self, seg: srf.Builder, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        stream = input_stream[0]

        def node_fn(input: srf.Observable, output: srf.Subscriber):

            producer = ck.Producer(self._kafka_conf)

            outstanding_requests = 0

            def on_next(x: typing.List[str]):
                nonlocal outstanding_requests

                def cb(_, msg):
                    if msg is not None and msg.value() is not None:
                        # fut.set_result(None)
                        pass
                    else:
                        # fut.set_exception(err or msg.error())
                        logger.error(("Error occurred in `to-kafka` stage with broker '%s' "
                                      "while committing message:\n%s\nError:\n%s"),
                                     self._kafka_conf["bootstrap.servers"],
                                     msg.value(),
                                     msg.error())
                        output.on_error(msg.error())

                records = serializers.df_to_json(x.df, strip_newlines=True)
                for m in records:

                    # Push all of the messages
                    while True:
                        try:
                            # this runs asynchronously, in C-K's thread
                            producer.produce(self._output_topic, m, callback=cb)
                            break
                        except BufferError:
                            time.sleep(self._poll_time)
                        except Exception:
                            logger.exception(("Error occurred in `to-kafka` stage with broker '%s' "
                                              "while committing message:\n%s"),
                                             self._kafka_conf["bootstrap.servers"],
                                             m)
                            break
                        finally:
                            # Try and process some
                            producer.poll(0)

                while len(producer) > 0:
                    producer.poll(0)

                return x

            def on_completed():

                producer.flush(-1)

            input.pipe(ops.map(on_next), ops.on_completed(on_completed)).subscribe(output)

            assert outstanding_requests == 0, "Not all inference requests were completed"

        # Write to kafka
        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        # node.launch_options.pe_count = self._max_concurrent

        # Return input unchanged
        return input_stream
