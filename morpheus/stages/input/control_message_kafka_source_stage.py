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

import logging
from io import StringIO

import mrc

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_control import MessageControl
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.kafka_source_stage import AutoOffsetReset
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage

logger = logging.getLogger(__name__)


@register_stage("from-cm-kafka", modes=[PipelineModes.AE])
class ControlMessageKafkaSourceStage(KafkaSourceStage):
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

        super().__init__(c,
                         bootstrap_servers,
                         input_topic,
                         group_id,
                         client_id,
                         poll_interval,
                         disable_commit,
                         disable_pre_filtering,
                         auto_offset_reset,
                         stop_after,
                         async_commits)

    @property
    def name(self) -> str:
        return "from-cm-kafka"

    def supports_cpp_node(self):
        return False

    def _convert_to_df(self, buffer: StringIO) -> cudf.DataFrame:

        df = super()._convert_to_df(buffer, engine="pandas", lines=True, orient="records")

        return df

    def _source_generator(self):

        source_gen = super()._source_generator()

        for message_meta in source_gen:

            df = message_meta.df

            if "inputs" not in df.columns:
                error_msg = "\nDataframe didn't have the required column `inputs`. Check the control message format."
                logger.error(error_msg)

                continue

            num_rows = len(df)

            # Iterate over each row in a dataframe.
            for i in range(num_rows):
                msg_inputs = df.inputs.iloc[i]
                # Iterate on inputs list to generate a control message.
                for msg_input in msg_inputs:
                    yield MessageControl(msg_input)

    def _build_source(self, builder: mrc.Builder) -> StreamPair:

        source = builder.make_source(self.unique_name, self._source_generator)

        return source, MessageControl
