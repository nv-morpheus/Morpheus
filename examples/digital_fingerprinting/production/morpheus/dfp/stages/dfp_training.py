# Copyright (c) 2022, NVIDIA CORPORATION.
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
import typing

import srf
from dfencoder import AutoEncoder
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.user_model_manager import UserModelManager

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTraining(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)

        self._user_models: typing.Dict[str, UserModelManager] = {}

    @property
    def name(self) -> str:
        return "dfp-training"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiDFPMessage, )

    def on_data(self, message: MultiDFPMessage):
        if (message is None or message.mess_count == 0):
            return None

        user_id = message.user_id
        model_manager = UserModelManager(self._config,
                                         user_id=user_id,
                                         save_model=False,
                                         epochs=30,
                                         min_history=300,
                                         max_history=-1,
                                         seed=42,
                                         model_class=AutoEncoder)

        model = model_manager.train(message.get_meta_dataframe())

        output_message = MultiAEMessage(message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=model)

        return output_message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiAEMessage
