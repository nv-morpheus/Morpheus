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
import time
import typing

import mrc
from mlflow.tracking.client import MlflowClient

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.model_cache import ModelCache
from ..utils.model_cache import ModelManager

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPInferenceStage(SinglePortStage):

    def __init__(self, c: Config, model_name_formatter: str = "dfp-{user_id}"):
        super().__init__(c)

        self._client = MlflowClient()
        self._fallback_user = self._config.ae.fallback_username

        self._model_cache: typing.Dict[str, ModelCache] = {}
        self._model_cache_size_max = 10

        self._cache_timeout_sec = 600

        self._model_manager = ModelManager(model_name_formatter=model_name_formatter)

    @property
    def name(self) -> str:
        return "dfp-inference"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiDFPMessage, )

    def get_model(self, user: str) -> ModelCache:

        return self._model_manager.load_user_model(self._client, user_id=user, fallback_user_ids=[self._fallback_user])

    def on_data(self, message: MultiDFPMessage):
        if (not message or message.mess_count == 0):
            return None

        start_time = time.time()

        df_user = message.get_meta()
        user_id = message.user_id

        try:
            model_cache = self.get_model(user_id)

            if (model_cache is None):
                raise RuntimeError("Could not find model for user {}".format(user_id))

            loaded_model = model_cache.load_model(self._client)

        except Exception:  # TODO
            logger.exception("Error trying to get model")
            return None

        post_model_time = time.time()

        results_df = loaded_model.get_results(df_user, return_abs=True)

        # Create an output message to allow setting meta
        output_message = MultiAEMessage(message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=loaded_model)

        output_message.set_meta(list(results_df.columns), results_df)

        output_message.set_meta('model_version', f"{model_cache.reg_model_name}:{model_cache.reg_model_version}")

        if logger.isEnabledFor(logging.DEBUG):
            load_model_duration = (post_model_time - start_time) * 1000.0
            get_anomaly_duration = (time.time() - post_model_time) * 1000.0

            logger.debug("Completed inference for user %s. Model load: %s ms, Model infer: %s ms. Start: %s, End: %s",
                         user_id,
                         load_model_duration,
                         get_anomaly_duration,
                         df_user[self._config.ae.timestamp_column_name].min(),
                         df_user[self._config.ae.timestamp_column_name].max())

        return output_message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        # node.launch_options.pe_count = self._config.num_threads

        return node, MultiAEMessage
