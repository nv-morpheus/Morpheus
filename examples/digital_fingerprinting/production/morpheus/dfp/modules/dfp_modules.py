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
import os
import pickle
import time
import typing
import uuid
from contextlib import contextmanager

import numpy as np
import pandas as pd
import srf
from dfencoder import AutoEncoder
from dfp.utils.cached_user_window import CachedUserWindow
from dfp.utils.logging_timer import log_time
from srf.core import operators as ops

import cudf

from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.modules.file_batcher_module import make_file_batcher_module
from morpheus.modules.file_to_df_module import make_file_to_df_module
from morpheus.modules.mlflow_model_writer_module import make_mlflow_model_writer_module
from morpheus.modules.nested_module import make_nested_module
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.decorators import register_module

from ..messages.multi_dfp_message import DFPMessageMeta
from ..messages.multi_dfp_message import MultiDFPMessage

logger = logging.getLogger(f"morpheus.{__name__}")


def make_dfp_training_module(module_id: str, namespace: str):
    """
    This function creates a DFP training module and registers it in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    """

    @register_module(module_id, namespace)
    def module_init(builder: srf.Builder):

        config = builder.get_current_module_config()

        if module_id in config:
            config = config[module_id]

        def on_data(message: MultiDFPMessage):
            if (message is None or message.mess_count == 0):
                return None

            user_id = message.user_id

            model = AutoEncoder(**config["model_kwargs"])

            final_df = message.get_meta_dataframe()

            # Only train on the feature columns
            final_df = final_df[final_df.columns.intersection(config["feature_columns"])]

            logger.debug("Training AE model for user: '%s'...", user_id)
            model.fit(final_df, epochs=30)
            logger.debug("Training AE model for user: '%s'... Complete.", user_id)

            output_message = MultiAEMessage(message.meta,
                                            mess_offset=message.mess_offset,
                                            mess_count=message.mess_count,
                                            model=model)

            return output_message

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        node = builder.make_node_full(str(uuid.uuid4()), node_fn)

        builder.register_module_input("input", node)
        builder.register_module_output("output", node)


def make_dfp_split_users_module(module_id: str, namespace: str):
    """
    This function creates a DFP split users module and registers it in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    """

    @register_module(module_id, namespace)
    def module_init(builder: srf.Builder):
        config = builder.get_current_module_config()

        if module_id in config:
            config = config[module_id]

        skip_users = config["skip_users"] if config["skip_users"] is not None else []
        only_users = config["only_users"] if config["only_users"] is not None else []

        # Map of user ids to total number of messages. Keeps indexes monotonic and increasing per user
        user_index_map: typing.Dict[str, int] = {}

        def extract_users(message: cudf.DataFrame):
            if (message is None):
                return []

            with log_time(logger.debug) as log_info:

                if (isinstance(message, cudf.DataFrame)):
                    # Convert to pandas because cudf is slow at this
                    message = message.to_pandas()

                split_dataframes: typing.Dict[str, cudf.DataFrame] = {}

                # If we are skipping users, do that here
                if (len(skip_users) > 0):
                    message = message[~message[config["userid_column_name"]].isin(skip_users)]

                if (len(only_users) > 0):
                    message = message[message[config["userid_column_name"]].isin(only_users)]

                # Split up the dataframes
                if (config["include_generic"]):
                    split_dataframes[config["fallback_username"]] = message

                if (config["include_individual"]):

                    split_dataframes.update(
                        {username: user_df
                         for username, user_df in message.groupby("username", sort=False)})

                output_messages: typing.List[DFPMessageMeta] = []

                for user_id in sorted(split_dataframes.keys()):

                    if (user_id in skip_users):
                        continue

                    user_df = split_dataframes[user_id]

                    current_user_count = user_index_map.get(user_id, 0)

                    # Reset the index so that users see monotonically increasing indexes
                    user_df.index = range(current_user_count, current_user_count + len(user_df))
                    user_index_map[user_id] = current_user_count + len(user_df)

                    output_messages.append(DFPMessageMeta(df=user_df, user_id=user_id))

                rows_per_user = [len(x.df) for x in output_messages]

                if (len(output_messages) > 0):
                    log_info.set_log(
                        ("Batch split users complete. Input: %s rows from %s to %s. "
                         "Output: %s users, rows/user min: %s, max: %s, avg: %.2f. Duration: {duration:.2f} ms"),
                        len(message),
                        message[config["timestamp_column_name"]].min(),
                        message[config["timestamp_column_name"]].max(),
                        len(rows_per_user),
                        np.min(rows_per_user),
                        np.max(rows_per_user),
                        np.mean(rows_per_user),
                    )

                return output_messages

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(extract_users), ops.flatten()).subscribe(sub)

        node = builder.make_node_full(str(uuid.uuid4()), node_fn)

        builder.register_module_input("input", node)
        builder.register_module_output("output", node)


def make_dfp_rolling_window_module(module_id: str, namespace: str):
    """
    This function creates a DFP rolling window module and registers it in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    """

    @register_module(module_id, namespace)
    def module_init(builder: srf.Builder):

        config = builder.get_current_module_config()

        if module_id in config:
            config = config[module_id]

        cache_dir = os.path.join(config["cache_dir"], "rolling-user-data")
        user_cache_map: typing.Dict[str, CachedUserWindow] = {}

        @contextmanager
        def get_user_cache(user_id: str):

            # Determine cache location
            cache_location = os.path.join(cache_dir, f"{user_id}.pkl")

            user_cache = None

            user_cache = user_cache_map.get(user_id, None)

            if (user_cache is None):
                user_cache = CachedUserWindow(user_id=user_id,
                                              cache_location=cache_location,
                                              timestamp_column=config["timestamp_column_name"])

                user_cache_map[user_id] = user_cache

            yield user_cache

            # # When it returns, make sure to save
            # user_cache.save()

        def build_window(message: DFPMessageMeta) -> MultiDFPMessage:

            user_id = message.user_id

            with get_user_cache(user_id) as user_cache:

                incoming_df = message.get_df()
                # existing_df = user_cache.df

                if (not user_cache.append_dataframe(incoming_df=incoming_df)):
                    # Then our incoming dataframe wasnt even covered by the window. Generate warning
                    logger.warn(("Incoming data preceeded existing history. "
                                 "Consider deleting the rolling window cache and restarting."))
                    return None

                # Exit early if we dont have enough data
                if (user_cache.count < config["min_history"]):
                    return None

                # We have enough data, but has enough time since the last training taken place?
                if (user_cache.total_count - user_cache.last_train_count < config["min_increment"]):
                    return None

                # Save the last train statistics
                train_df = user_cache.get_train_df(max_history=config["max_history"])

                # Hash the incoming data rows to find a match
                incoming_hash = pd.util.hash_pandas_object(incoming_df.iloc[[0, -1]], index=False)

                # Find the index of the first and last row
                match = train_df[train_df["_row_hash"] == incoming_hash.iloc[0]]

                if (len(match) == 0):
                    raise RuntimeError("Invalid rolling window")

                first_row_idx = match.index[0].item()
                last_row_idx = train_df[train_df["_row_hash"] == incoming_hash.iloc[-1]].index[-1].item()

                found_count = (last_row_idx - first_row_idx) + 1

                if (found_count != len(incoming_df)):
                    raise RuntimeError(("Overlapping rolling history detected. "
                                        "Rolling history can only be used with non-overlapping batches"))

                train_offset = train_df.index.get_loc(first_row_idx)

                # Otherwise return a new message
                return MultiDFPMessage(meta=DFPMessageMeta(df=train_df, user_id=user_id),
                                       mess_offset=train_offset,
                                       mess_count=found_count)

        def on_data(message: DFPMessageMeta):

            with log_time(logger.debug) as log_info:

                result = build_window(message)

                if (result is not None):

                    log_info.set_log(
                        ("Rolling window complete for %s in {duration:0.2f} ms. "
                         "Input: %s rows from %s to %s. Output: %s rows from %s to %s"),
                        message.user_id,
                        len(message.df),
                        message.df[config["timestamp_column_name"]].min(),
                        message.df[config["timestamp_column_name"]].max(),
                        result.mess_count,
                        result.get_meta(config["timestamp_column_name"]).min(),
                        result.get_meta(config["timestamp_column_name"]).max(),
                    )
                else:
                    # Dont print anything
                    log_info.disable()

                return result

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        node = builder.make_node_full(str(uuid.uuid4()), node_fn)

        builder.register_module_input("input", node)
        builder.register_module_output("output", node)


def make_dfp_preporcessing_module(module_id: str, namespace: str):
    """
    This function creates a DFP preprocessing module and registers it in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    """

    @register_module(module_id, namespace)
    def module_init(builder: srf.Builder):

        config = builder.get_current_module_config()

        if module_id in config:
            config = config[module_id]

        if "schema" not in config:
            raise Exception("Preprocess schema doesn't exist")

        schema_config = config["schema"]

        schema = pickle.loads(bytes(schema_config["schema_str"], schema_config["encoding"]))

        def process_features(message: MultiDFPMessage):
            if (message is None):
                return None

            start_time = time.time()

            # Process the columns
            df_processed = process_dataframe(message.get_meta_dataframe(), schema)

            # Apply the new dataframe, only the rows in the offset
            message.set_meta_dataframe(list(df_processed.columns), df_processed)

            if logger.isEnabledFor(logging.DEBUG):
                duration = (time.time() - start_time) * 1000.0

                logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                             message.mess_count,
                             message.get_meta(config["timestamp_column_name"]).min(),
                             message.get_meta(config["timestamp_column_name"]).max(),
                             duration)

            return message

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(process_features)).subscribe(sub)

        node = builder.make_node_full(str(uuid.uuid4()), node_fn)

        builder.register_module_input("input", node)
        builder.register_module_output("output", node)


def make_modules():
    namespace = "morpheus_modules"

    # Register file batcher module
    make_file_batcher_module("FileBatcher", namespace)

    # Register file to dataframe module
    make_file_to_df_module("FileToDataFrame", namespace)

    # Register DFP split data by users module
    make_dfp_split_users_module("DFPSplitUsers", namespace)

    # Register DFP rolling window module
    make_dfp_rolling_window_module("DFPRollingWindow", namespace)

    # Register DFP processing module
    make_dfp_preporcessing_module("DFPPreprocessing", namespace)

    # register DFP training module
    make_dfp_training_module("DFPTraining", namespace)

    # Register MLFlow model writer module
    make_mlflow_model_writer_module("MLFlowModelWriter", namespace)

    # Ordered DFP pipeline preprocessing modules meta
    ordered_proc_modules_meta = [("FileBatcher", namespace), ("FileToDataFrame", namespace),
                                 ("DFPSplitUsers", namespace), ("DFPRollingWindow", namespace),
                                 ("DFPPreprocessing", namespace)]

    # Ordered DFP pipeline training modules meta
    ordered_train_modules_meta = [("DFPTraining", namespace), ("MLFlowModelWriter", namespace)]

    # Register DFP preprocessing pipeline module
    make_nested_module(module_id="DFPPipelinePreprocessing",
                       namespace=namespace,
                       ordered_modules_meta=ordered_proc_modules_meta)

    # Register DFP training pipeline module
    make_nested_module(module_id="DFPPipelineTraining",
                       namespace=namespace,
                       ordered_modules_meta=ordered_train_modules_meta)
