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
import time

import mrc
import cudf
from dfp.utils.model_cache import ModelCache
from dfp.utils.model_cache import ModelManager
from mlflow.tracking.client import MlflowClient
from mrc.core import operators as ops

from ..messages.multi_dfp_message import MultiDFPMessage, DFPMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.messages import MessageControl
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.module_ids import DFP_INFERENCE

logger = logging.getLogger(__name__)


@register_module(DFP_INFERENCE, MODULE_NAMESPACE)
def dfp_inference(builder: mrc.Builder):
    """
    Inference module function.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_INFERENCE, builder)

    fallback_user = config.get("fallback_username", None)
    model_name_formatter = config.get("model_name_formatter", None)
    timestamp_column_name = config.get("timestamp_column_name", None)

    client = MlflowClient()
    model_manager = ModelManager(model_name_formatter=model_name_formatter)

    def get_model(user: str) -> ModelCache:

        return model_manager.load_user_model(client, user_id=user, fallback_user_ids=[fallback_user])

    def process_task(control_message: MessageControl, task: dict):
        start_time = time.time()

        task_params = task['properties']
        data_source = task_params['data']
        if (data_source != "payload"):
            raise ValueError("Unsupported data source: {}".format(data_source))

        user_id = task_params['user_id']

        payload = control_message.payload()
        df_user = payload.df.to_pandas()

        try:
            model_cache: ModelCache = get_model(user_id)

            if (model_cache is None):
                raise RuntimeError("Could not find model for user {}".format(user_id))

            loaded_model = model_cache.load_model(client)

        except Exception:  # TODO
            logger.exception("Error trying to get model")
            return None

        post_model_time = time.time()

        results_df = loaded_model.get_results(df_user, return_abs=True)

        # Create an output message to allow setting meta
        dfp_mm = DFPMessageMeta(results_df, user_id=user_id)
        multi_message = MultiDFPMessage(dfp_mm, mess_offset=0, mess_count=len(results_df))
        output_message = MultiAEMessage(multi_message.meta,
                                        mess_offset=multi_message.mess_offset,
                                        mess_count=multi_message.mess_count,
                                        model=loaded_model)

        output_message.set_meta(list(results_df.columns), results_df)

        output_message.set_meta('model_version', f"{model_cache.reg_model_name}:{model_cache.reg_model_version}")

        if logger.isEnabledFor(logging.DEBUG):
            load_model_duration = (post_model_time - start_time) * 1000.0
            get_anomaly_duration = (time.time() - post_model_time) * 1000.0

            logger.debug(
                "Completed inference for user %s. Model load: %s ms, Model infer: %s ms. Start: %s, End: %s",
                user_id,
                load_model_duration,
                get_anomaly_duration,
                df_user[timestamp_column_name].min(),
                df_user[timestamp_column_name].max())

        return output_message

    def on_data(control_message: MessageControl):
        config = control_message.config()
        for task in config['tasks']:
            if task['type'] == 'inference':
                # TODO(Devin): Decide on what to do if we have multiple inference tasks
                process_task(control_message, task)

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data)).subscribe(sub)

    node = builder.make_node_full(DFP_INFERENCE, node_fn)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
