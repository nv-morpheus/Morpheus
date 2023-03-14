# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import pandas as pd
from dfp.utils.model_cache import ModelCache
from dfp.utils.model_cache import ModelManager
from mlflow.tracking.client import MlflowClient
from mrc.core import operators as ops

from morpheus.messages import MessageControl
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import register_module

from ..messages.multi_dfp_message import DFPMessageMeta
from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.module_ids import DFP_INFERENCE

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_INFERENCE, MORPHEUS_MODULE_NAMESPACE)
def dfp_inference(builder: mrc.Builder):
    """
    Inference module function.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.

    Notes
    ----------
    Configurable parameters:
        - model_name_formatter: Formatter for model names
        - fallback_username: Fallback user to use if no model is found for a user
        - timestamp_column_name: Name of the timestamp column
    """

    config = builder.get_current_module_config()

    if ("model_name_formatter" not in config):
        raise ValueError("Inference module requires model_name_formatter to be configured")

    if ("fallback_username" not in config):
        raise ValueError("Inference module requires fallback_username to be configured")

    model_name_formatter = config.get("model_name_formatter", None)
    fallback_user = config.get("fallback_username", "generic_user")

    timestamp_column_name = config.get("timestamp_column_name", "timestamp")

    client = MlflowClient()
    model_manager = ModelManager(model_name_formatter=model_name_formatter)

    def get_model(user: str) -> ModelCache:

        return model_manager.load_user_model(client, user_id=user, fallback_user_ids=[fallback_user])

    def process_task(control_message: MessageControl, task: dict):
        start_time = time.time()

        user_id = control_message.get_metadata("user_id")
        payload = control_message.payload()
        with payload.mutable_dataframe() as dfm:
            df_user = dfm.to_pandas()
        df_user[timestamp_column_name] = pd.to_datetime(df_user[timestamp_column_name], utc=True)

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

        include_cols = set(df_user.columns) - set(results_df.columns)

        for col in include_cols:
            results_df[col] = df_user[col].copy(True)

        # Create an output message to allow setting meta
        dfp_mm = DFPMessageMeta(results_df, user_id=user_id)
        multi_message = MultiDFPMessage(dfp_mm, mess_offset=0, mess_count=len(results_df))

        output_message = MultiAEMessage(multi_message.meta,
                                        multi_message.mess_offset,
                                        multi_message.mess_count,
                                        loaded_model)
        # output_message.set_meta(list(results_df.columns), results_df)
        output_message.set_meta('model_version', f"{model_cache.reg_model_name}:{model_cache.reg_model_version}")

        if logger.isEnabledFor(logging.DEBUG):
            load_model_duration = (post_model_time - start_time) * 1000.0
            get_anomaly_duration = (time.time() - post_model_time) * 1000.0

            logger.debug("Completed inference for user %s. Model load: %s ms, Model infer: %s ms. Start: %s, End: %s",
                         user_id,
                         load_model_duration,
                         get_anomaly_duration,
                         df_user[timestamp_column_name].min(),
                         df_user[timestamp_column_name].max())

        return output_message

    def on_data(control_message: MessageControl):
        if (control_message is None):
            return None

        task_results = []
        while (control_message.has_task("inference")):
            task = control_message.pop_task("inference")
            task_results.append(process_task(control_message, task))

        return task_results

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.flatten()).subscribe(sub)

    node = builder.make_node(DFP_INFERENCE, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
