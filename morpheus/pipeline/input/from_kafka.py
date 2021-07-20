# Copyright (c) 2021, NVIDIA CORPORATION.
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

import typing

from streamz import Source
from streamz.core import Stream
from tornado.ioloop import IOLoop

import cudf

from morpheus.config import Config
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture


class KafkaSourceStage(SingleOutputSource):
    """
    Load messages from a Kafka cluster.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    bootstrap_servers : str
        Kafka cluster bootstrap servers separated by a comma.
    input_topic : str
        Input kafka topic
    group_id : str
        Specifies the name of the consumer group a Kafka consumer belongs to
    use_dask : bool
        Determines whether or not dask should be used to consume messages. Operates independently of the
        `Pipeline.use_dask` option
    poll_interval : str
        Seconds that elapse between polling Kafka for new messages. Follows the pandas interval format

    """
    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "custreamz",
                 use_dask: bool = False,
                 poll_interval: str = "10millis"):
        super().__init__(c)

        self._consumer_conf = {
            'bootstrap.servers': bootstrap_servers, 'group.id': group_id, 'session.timeout.ms': "60000"
        }

        self._input_topic = input_topic
        self._use_dask = use_dask
        self._poll_interval = poll_interval
        self._max_batch_size = c.pipeline_batch_size
        self._client = None

    @property
    def name(self) -> str:
        return "from-kafka"

    def _build_source(self) -> typing.Tuple[Source, typing.Type]:

        if (self._use_dask):
            from dask.distributed import Client
            self._client = Client()

            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=True,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

            return source, StreamFuture[cudf.DataFrame]
        else:
            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=False,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

            return source, cudf.DataFrame
