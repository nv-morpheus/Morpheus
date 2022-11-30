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

from contextlib import contextmanager
import datetime
from functools import partial
import hashlib
import logging
import os
import random
import time
import typing
import urllib.parse
from dfp.stages.dfp_file_batcher_stage import TimestampFileObj
from dfp.utils.file_utils import date_extractor, iso_date_regex
import cudf
from dfp.utils.column_info import BoolColumn, create_increment_col, process_dataframe
from dfp.utils.column_info import ColumnInfo
from dfp.utils.column_info import CustomColumn
from dfp.utils.column_info import DataFrameInputSchema
from dfp.utils.column_info import DateTimeColumn
from dfp.utils.column_info import IncrementColumn
from dfp.utils.column_info import RenameColumn
from dfp.utils.column_info import StringCatColumn
from dfp.stages.dfp_rolling_window_stage import CachedUserWindow
import numpy as np
import dask
from dask.distributed import Client
from dask.distributed import LocalCluster
from dfp.utils.logging_timer import log_time
import mlflow
import requests
from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
import srf
import fsspec
import fsspec.utils
import pandas as pd
from dfencoder import AutoEncoder
from dfp.utils.model_cache import user_to_model_name
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec
from mlflow.types import Schema
from mlflow.types.utils import _infer_pandas_column
from mlflow.types.utils import _infer_schema
from srf.core import operators as ops
import json
import multiprocessing as mp

from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.utils.decorator_utils import is_module_registered

from ..messages.multi_dfp_message import DFPMessageMeta, MultiDFPMessage

logger = logging.getLogger(f"morpheus.{__name__}")


class DFPModuleRegisterUtil:

    def __init__(self):
        self._registry = srf.ModuleRegistry
        self._release_version = self.get_release_version()

    def get_release_version(self) -> typing.List[int]:
        ver_list = srf.__version__.split('.')
        ver_list = [int(i) for i in ver_list]
        return ver_list

    def register_training_module(self, module_id, namespace):
        
        def training_module(builder: srf.Builder):

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

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, training_module)

    def register_mlflow_model_writer_module(self, module_id, namespace):

        def mlflow_writer_module(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            def user_id_to_model(user_id: str):

                return user_to_model_name(user_id=user_id, model_name_formatter=config["model_name_formatter"])

            def user_id_to_experiment(user_id: str):
                kwargs = {
                    "user_id": user_id,
                    "user_md5": hashlib.md5(user_id.encode('utf-8')).hexdigest(),
                    "reg_model_name": user_id_to_model(user_id=user_id)
                }

                return config["experiment_name_formatter"].format(**kwargs)

            def _apply_model_permissions(reg_model_name: str):

                # Check the required variables
                databricks_host = os.environ.get("DATABRICKS_HOST", None)
                databricks_token = os.environ.get("DATABRICKS_TOKEN", None)

                if (databricks_host is None or databricks_token is None):
                    raise RuntimeError("Cannot set Databricks model permissions. "
                                       "Environment variables `DATABRICKS_HOST` and `DATABRICKS_TOKEN` must be set")

                headers = {"Authorization": f"Bearer {databricks_token}"}

                url_base = f"{databricks_host}"

                try:
                    # First get the registered model ID
                    get_registered_model_url = urllib.parse.urljoin(url_base,
                                                                    "/api/2.0/mlflow/databricks/registered-models/get")

                    get_registered_model_response = requests.get(url=get_registered_model_url,
                                                                 headers=headers,
                                                                 params={"name": reg_model_name})

                    registered_model_response = get_registered_model_response.json()

                    reg_model_id = registered_model_response["registered_model_databricks"]["id"]

                    # Now apply the permissions. If it exists already, it will be overwritten or it is a no-op
                    patch_registered_model_permissions_url = urllib.parse.urljoin(
                        url_base, f"/api/2.0/preview/permissions/registered-models/{reg_model_id}")

                    patch_registered_model_permissions_body = {
                        "access_control_list": [{
                            "group_name": group, "permission_level": permission
                        } for group,
                                                permission in config["databricks_permissions"].items()]
                    }

                    requests.patch(url=patch_registered_model_permissions_url,
                                   headers=headers,
                                   json=patch_registered_model_permissions_body)

                except Exception:
                    logger.exception("Error occurred trying to apply model permissions to model: %s",
                                     reg_model_name,
                                     exc_info=True)

            def on_data(message: MultiAEMessage):

                user = message.meta.user_id

                model: AutoEncoder = message.model

                model_path = "dfencoder"
                reg_model_name = user_id_to_model(user_id=user)

                # Write to ML Flow
                try:
                    mlflow.end_run()

                    experiment_name = user_id_to_experiment(user_id=user)

                    # Creates a new experiment if it doesnt exist
                    experiment = mlflow.set_experiment(experiment_name)

                    with mlflow.start_run(run_name="Duo autoencoder model training run",
                                          experiment_id=experiment.experiment_id) as run:

                        model_path = f"{model_path}-{run.info.run_uuid}"

                        # Log all params in one dict to avoid round trips
                        mlflow.log_params({
                            "Algorithm": "Denosing Autoencoder",
                            "Epochs": model.lr_decay.state_dict().get("last_epoch", "unknown"),
                            "Learning rate": model.lr,
                            "Batch size": model.batch_size,
                            "Start Epoch": message.get_meta("timestamp").min(),
                            "End Epoch": message.get_meta("timestamp").max(),
                            "Log Count": message.mess_count,
                        })

                        metrics_dict: typing.Dict[str, float] = {}

                        # Add info on the embeddings
                        for k, v in model.categorical_fts.items():
                            embedding = v.get("embedding", None)

                            if (embedding is None):
                                continue

                            metrics_dict[f"embedding-{k}-num_embeddings"] = embedding.num_embeddings
                            metrics_dict[f"embedding-{k}-embedding_dim"] = embedding.embedding_dim

                        # Add metrics for all of the loss stats
                        if (hasattr(model, "feature_loss_stats")):
                            for k, v in model.feature_loss_stats.items():
                                metrics_dict[f"loss-{k}-mean"] = v.get("mean", "unknown")
                                metrics_dict[f"loss-{k}-std"] = v.get("std", "unknown")

                        mlflow.log_metrics(metrics_dict)

                        # Use the prepare_df function to setup the direct inputs to the model. Only include features
                        # returned by prepare_df to show the actual inputs to the model (any extra are discarded)
                        input_df = message.get_meta().iloc[0:1]
                        prepared_df = model.prepare_df(input_df)
                        output_values = model.get_anomaly_score(input_df)

                        input_schema = Schema([
                            ColSpec(type=_infer_pandas_column(input_df[col_name]), name=col_name)
                            for col_name in list(prepared_df.columns)
                        ])
                        output_schema = _infer_schema(output_values)

                        model_sig = ModelSignature(inputs=input_schema, outputs=output_schema)

                        model_info = mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=model_path,
                            conda_env=config["conda_env"],
                            signature=model_sig,
                        )

                        client = MlflowClient()

                        # First ensure a registered model has been created
                        try:
                            create_model_response = client.create_registered_model(reg_model_name)
                            logger.debug("Successfully registered model '%s'.", create_model_response.name)
                        except MlflowException as e:
                            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                                pass
                            else:
                                raise e

                        # If we are using databricks, make sure we set the correct permissions
                        if (config["databricks_permissions"] is not None and mlflow.get_tracking_uri() == "databricks"):
                            # Need to apply permissions
                            _apply_model_permissions(reg_model_name=reg_model_name)

                        model_src = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)

                        tags = {
                            "start": message.get_meta(config["timestamp_column_name"]).min(),
                            "end": message.get_meta(config["timestamp_column_name"]).max(),
                            "count": message.get_meta(config["timestamp_column_name"]).count()
                        }

                        # Now create the model version
                        mv = client.create_model_version(name=reg_model_name,
                                                         source=model_src,
                                                         run_id=run.info.run_id,
                                                         tags=tags)

                        logger.debug("ML Flow model upload complete: %s:%s:%s", user, reg_model_name, mv.version)

                except Exception:
                    logger.exception("Error uploading model to ML Flow", exc_info=True)

                return message

            def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

            node = builder.make_node_full("dfp_mlflow_writer", node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, mlflow_writer_module)

    @is_module_registered
    def is_module_registered(self, module_id=None, namespace=None):
        return

    def register_chain_modules(self, module_id, namespace, ordered_module_meta):

        def module_init(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            prev_module = None
            head_module = None

            for item in ordered_module_meta:
                self.is_module_registered(module_id=item["module_id"], namespace=item["namespace"])

                curr_module = builder.load_module(item["module_id"], item["namespace"], "dfp_pipeline", config)

                if prev_module:
                    builder.make_edge(prev_module.output_port("output"), curr_module.input_port("input"))
                else:
                    head_module = curr_module

                prev_module = curr_module

            builder.register_module_input("input", head_module.input_port("input"))
            builder.register_module_output("output", prev_module.output_port("output"))

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, module_init)

    def get_unique_name(self, id) -> str:
        unique_name = id + "-" + str(random.randint(0, 1000))
        return unique_name

    def register_file_batcher_module(self, module_id, namespace):

        def file_batcher_module(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            def on_data(file_objects: fsspec.core.OpenFiles):

                # Determine the date of the file, and apply the window filter if we have one
                ts_and_files = []
                for file_object in file_objects:
                    ts = date_extractor(file_object, iso_date_regex)

                    # Exclude any files outside the time window
                    if ((config["start_time"] is not None and ts < config["start_time"])
                            or (config["end_time"] is not None and ts > config["end_time"])):
                        continue

                    ts_and_files.append(TimestampFileObj(ts, file_object))

                # sort the incoming data by date
                ts_and_files.sort(key=lambda x: x.timestamp)

                # Create a dataframe with the incoming metadata
                if ((len(ts_and_files) > 1) and (config["sampling_rate_s"] > 0)):
                    file_sampled_list = []

                    ts_last = ts_and_files[0].timestamp

                    file_sampled_list.append(ts_and_files[0])

                    for idx in range(1, len(ts_and_files)):
                        ts = ts_and_files[idx].timestamp

                        if ((ts - ts_last).seconds >= config["sampling_rate_s"]):

                            ts_and_files.append(ts_and_files[idx])
                            ts_last = ts
                    else:
                        ts_and_files = file_sampled_list

                df = pd.DataFrame()

                timestamps = []
                full_names = []
                file_objs = []
                for (ts, file_object) in ts_and_files:
                    timestamps.append(ts)
                    full_names.append(file_object.full_name)
                    file_objs.append(file_object)

                df["dfp_timestamp"] = timestamps
                df["key"] = full_names
                df["objects"] = file_objs

                output_batches = []

                if len(df) > 0:
                    # Now split by the batching settings
                    df_period = df["dfp_timestamp"].dt.to_period(config["period"])

                    period_gb = df.groupby(df_period)

                    n_groups = len(period_gb)
                    for group in period_gb.groups:
                        period_df = period_gb.get_group(group)

                        obj_list = fsspec.core.OpenFiles(period_df["objects"].to_list(),
                                                         mode=file_objects.mode,
                                                         fs=file_objects.fs)

                        output_batches.append((obj_list, n_groups))

                return output_batches

            def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                obs.pipe(ops.map(on_data), ops.flatten()).subscribe(sub)

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, file_batcher_module)

    def register_file_to_dataframe_module(self, module_id, namespace):

        def file_to_dataframe_module(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            download_method: typing.Literal["single_thread", "multiprocess", "dask",
                                            "dask_thread"] = os.environ.get("MORPHEUS_FILE_DOWNLOAD_TYPE",
                                                                            "dask_thread")
            cache_dir = os.path.join(config["cache_dir"], "file_cache")

            dask_cluster = None

            source_column_info = [
                DateTimeColumn(name=config["timestamp_column_name"], dtype=datetime, input_name="timestamp"),
                RenameColumn(name=config["userid_column_name"], dtype=str, input_name="user.name"),
                RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
                RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
                StringCatColumn(name="location",
                                dtype=str,
                                input_columns=[
                                    "access_device.location.city",
                                    "access_device.location.state",
                                    "access_device.location.country"
                                ],
                                sep=", "),
                RenameColumn(name="authdevicename", dtype=str, input_name="auth_device.name"),
                BoolColumn(name="result",
                           dtype=bool,
                           input_name="result",
                           true_values=["success", "SUCCESS"],
                           false_values=["denied", "DENIED", "FRAUD"]),
                ColumnInfo(name="reason", dtype=str),
            ]

            schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                          column_info=source_column_info)

            def get_dask_cluster():

                if (dask_cluster is None):
                    logger.debug("Creating dask cluster...")

                    # Up the heartbeat interval which can get violated with long download times
                    dask.config.set({"distributed.client.heartbeat": "30s"})

                    dask_cluster = LocalCluster(start=True, processes=not download_method == "dask_thread")

                    logger.debug("Creating dask cluster... Done. Dashboard: %s", dask_cluster.dashboard_link)

                return dask_cluster

            def close_dask_cluster():
                if (dask_cluster is not None):
                    logger.debug("Stopping dask cluster...")

                    dask_cluster.close()

                    dask_cluster = None

                    logger.debug("Stopping dask cluster... Done.")

            def single_object_to_dataframe(file_object: fsspec.core.OpenFile,
                                           schema: DataFrameInputSchema,
                                           file_type: FileTypes,
                                           filter_null: bool,
                                           parser_kwargs: dict):

                retries = 0
                s3_df = None
                while (retries < 2):
                    try:
                        with file_object as f:
                            s3_df = read_file_to_df(f,
                                                    file_type,
                                                    filter_nulls=filter_null,
                                                    df_type="pandas",
                                                    parser_kwargs=parser_kwargs)

                        break
                    except Exception as e:
                        if (retries < 2):
                            logger.warning("Refreshing S3 credentials")
                            # cred_refresh()
                            retries += 1
                        else:
                            raise e

                # Run the pre-processing before returning
                if (s3_df is None):
                    return s3_df

                s3_df = process_dataframe(df_in=s3_df, input_schema=schema)

                return s3_df

            def get_or_create_dataframe_from_s3_batch(
                    file_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]) -> typing.Tuple[cudf.DataFrame, bool]:

                if (not file_object_batch):
                    return None, False

                file_list = file_object_batch[0]
                batch_count = file_object_batch[1]

                fs: fsspec.AbstractFileSystem = file_list.fs

                # Create a list of dictionaries that only contains the information we are interested in hashing. `ukey` just
                # hashes all of the output of `info()` which is perfect
                hash_data = [{"ukey": fs.ukey(file_object.path)} for file_object in file_list]

                # Convert to base 64 encoding to remove - values
                objects_hash_hex = hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

                batch_cache_location = os.path.join(cache_dir, "batches", f"{objects_hash_hex}.pkl")

                # Return the cache if it exists
                if (os.path.exists(batch_cache_location)):
                    output_df = pd.read_pickle(batch_cache_location)
                    output_df["origin_hash"] = objects_hash_hex
                    output_df["batch_count"] = batch_count

                    return (output_df, True)

                # Cache miss
                download_method_func = partial(single_object_to_dataframe,
                                               schema=schema,
                                               file_type=FileTypes.Auto,
                                               filter_null=config["filter_null"],
                                               parser_kwargs=config["parser_kwargs"])

                download_buckets = file_list

                # Loop over dataframes and concat into one
                try:
                    dfs = []
                    if (download_method.startswith("dask")):

                        # Create the client each time to ensure all connections to the cluster are closed (they can time out)
                        with Client(get_dask_cluster()) as client:
                            dfs = client.map(download_method_func, download_buckets)

                            dfs = client.gather(dfs)

                    elif (download_method == "multiprocessing"):
                        # Use multiprocessing here since parallel downloads are a pain
                        with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
                            dfs = p.map(download_method_func, download_buckets)
                    else:
                        # Simply loop
                        for s3_object in download_buckets:
                            dfs.append(download_method(s3_object))

                except Exception:
                    logger.exception("Failed to download logs. Error: ", exc_info=True)
                    return None, False

                if (not dfs):
                    logger.error("No logs were downloaded")
                    return None, False

                output_df: pd.DataFrame = pd.concat(dfs)

                # Finally sort by timestamp and then reset the index
                output_df.sort_values(by=["timestamp"], inplace=True)

                output_df.reset_index(drop=True, inplace=True)

                # Save dataframe to cache future runs
                os.makedirs(os.path.dirname(batch_cache_location), exist_ok=True)

                try:
                    output_df.to_pickle(batch_cache_location)
                except Exception:
                    logger.warning("Failed to save batch cache. Skipping cache for this batch.", exc_info=True)

                output_df["batch_count"] = batch_count
                output_df["origin_hash"] = objects_hash_hex

                return (output_df, False)

            def convert_to_dataframe(s3_object_batch: typing.Tuple[fsspec.core.OpenFiles, int]):
                if (not s3_object_batch):
                    return None

                start_time = time.time()

                try:

                    output_df, cache_hit = get_or_create_dataframe_from_s3_batch(s3_object_batch)

                    duration = (time.time() - start_time) * 1000.0

                    logger.debug("S3 objects to DF complete. Rows: %s, Cache: %s, Duration: %s ms",
                                 len(output_df),
                                 "hit" if cache_hit else "miss",
                                 duration)

                    return output_df
                except Exception:
                    logger.exception("Error while converting S3 buckets to DF.")
                    raise

            def node_fn(obs: srf.Observable, sub: srf.Subscriber):
                obs.pipe(ops.map(convert_to_dataframe), ops.on_completed(close_dask_cluster)).subscribe(sub)

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, file_to_dataframe_module)

    def register_split_users_module(self, module_id, namespace):

        def split_users_module(builder: srf.Builder):
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

                        # logger.debug("Emitting dataframe for user '%s'. Start: %s, End: %s, Count: %s",
                        #              user,
                        #              df_user[self._config.ae.timestamp_column_name].min(),
                        #              df_user[self._config.ae.timestamp_column_name].max(),
                        #              df_user[self._config.ae.timestamp_column_name].count())

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

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, split_users_module)

    def register_rolling_window_module(self, module_id, namespace):

        def rolling_window_module(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            cache_dir = os.path.join(config["cache_dir"], "rolling-user-data")
            user_cache_map: typing.Dict[str, CachedUserWindow] = {}

            def trim_dataframe(df: pd.DataFrame):

                if (config["max_history"] is None):
                    return df

                # See if max history is an int
                if (isinstance(config["max_history"], int)):
                    return df.tail(config["max_history"])

                # If its a string, then its a duration
                if (isinstance(config["max_history"], str)):
                    # Get the latest timestamp
                    latest = df[config["timestamp_column_name"]].max()

                    time_delta = pd.Timedelta(config["max_history"])

                    # Calc the earliest
                    earliest = latest - time_delta

                    return df[df['timestamp'] >= earliest]

                raise RuntimeError("Unsupported max_history")

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

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, rolling_window_module)

    def register_preporcessing_module(self, module_id, namespace):

        def preprocessing_module(builder: srf.Builder):

            config = builder.get_current_module_config()

            if module_id in config:
                config = config[module_id]

            # Preprocessing schema
            preprocess_column_info = [
                ColumnInfo(name=config["timestamp_column_name"], dtype=datetime),
                ColumnInfo(name=config["userid_column_name"], dtype=str),
                ColumnInfo(name="accessdevicebrowser", dtype=str),
                ColumnInfo(name="accessdeviceos", dtype=str),
                ColumnInfo(name="authdevicename", dtype=str),
                ColumnInfo(name="result", dtype=bool),
                ColumnInfo(name="reason", dtype=str),
                # Derived columns
                IncrementColumn(name="logcount",
                                dtype=int,
                                input_name=config["timestamp_column_name"],
                                groupby_column=config["userid_column_name"]),
                CustomColumn(name="locincrement",
                             dtype=int,
                             process_column_fn=partial(create_increment_col, column_name="location")),
            ]

            preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

            def process_features(message: MultiDFPMessage):
                if (message is None):
                    return None

                start_time = time.time()

                # Process the columns
                df_processed = process_dataframe(message.get_meta_dataframe(), preprocess_schema)

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

            node = builder.make_node_full(self.get_unique_name(module_id), node_fn)

            builder.register_module_input("input", node)
            builder.register_module_output("output", node)

        if not self._registry.contains(module_id, namespace):
            self._registry.register_module(module_id, namespace, self._release_version, preprocessing_module)


dfp_util = DFPModuleRegisterUtil()

dfp_util.register_file_batcher_module("DFPFileBatcher", "morpheus_modules")

dfp_util.register_file_to_dataframe_module("DFPFileToDataFrame", "morpheus_modules")

dfp_util.register_split_users_module("DFPSplitUsers", "morpheus_modules")

dfp_util.register_rolling_window_module("DFPRollingWindow", "morpheus_modules")

dfp_util.register_preporcessing_module("DFPPreprocessing", "morpheus_modules")

# Register DFP training module
dfp_util.register_training_module("DFPTraining", "morpheus_modules")

# Register DFP MLFlow model writer module
dfp_util.register_mlflow_model_writer_module("DFPMLFlowModelWriter", "morpheus_modules")

ordered_dfp_pipeline_preprocessing_module_meta = [{
    "module_id": "DFPFileBatcher", "namespace": "morpheus_modules"
}, {
    "module_id": "DFPFileToDataFrame", "namespace": "morpheus_modules"
}, {
    "module_id": "DFPSplitUsers", "namespace": "morpheus_modules"
}, {
    "module_id": "DFPRollingWindow", "namespace": "morpheus_modules"
}, {
    "module_id": "DFPPreprocessing", "namespace": "morpheus_modules"
}]

ordered_dfp_pipeline_training_module_meta = [{
    "module_id": "DFPTraining", "namespace": "morpheus_modules"
}, {
    "module_id": "DFPMLFlowModelWriter", "namespace": "morpheus_modules"
}]

dfp_util.register_chain_modules("DFPPipelinePreprocessing",
                                "morpheus_modules",
                                ordered_dfp_pipeline_preprocessing_module_meta)

# Register DFP training pipeline module
dfp_util.register_chain_modules("DFPPipelineTraining", "morpheus_modules", ordered_dfp_pipeline_training_module_meta)