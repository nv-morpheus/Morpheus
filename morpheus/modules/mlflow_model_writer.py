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

import hashlib
import logging
import os
import typing
import urllib.parse

import mlflow
import mrc
import requests
from dfencoder import AutoEncoder
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
from mrc.core import operators as ops

from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(MLFLOW_MODEL_WRITER, MODULE_NAMESPACE)
def mlflow_model_writer(builder: mrc.Builder):
    """
    This module uploads trained models to the mlflow server.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.
    """

    config = get_module_config(MLFLOW_MODEL_WRITER, builder)

    model_name_formatter = config.get("model_name_formatter", None)
    experiment_name_formatter = config.get("experiment_name_formatter", None)
    conda_env = config.get("conda_env", None)
    timestamp_column_name = config.get("timestamp_column_name", None)
    databricks_permissions = config.get("databricks_permissions", None)

    def user_id_to_model(user_id: str):

        kwargs = {
            "user_id": user_id,
            "user_md5": hashlib.md5(user_id.encode('utf-8')).hexdigest(),
        }

        return model_name_formatter.format(**kwargs)

    def user_id_to_experiment(user_id: str):

        kwargs = {
            "user_id": user_id,
            "user_md5": hashlib.md5(user_id.encode('utf-8')).hexdigest(),
            "reg_model_name": user_id_to_model(user_id=user_id)
        }

        return experiment_name_formatter.format(**kwargs)

    def apply_model_permissions(reg_model_name: str):

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
                                        permission in databricks_permissions.items()]
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
        df = message.meta.df

        print(df.columns, flush=True)

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

                mlflow.log_metrics(metrics_dict)

                # Use the prepare_df function to setup the direct inputs to the model. Only include features
                # returned by prepare_df to show the actual inputs to the model (any extra are discarded)
                input_df = message.get_meta().iloc[0:1].to_pandas()
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
                    conda_env=conda_env,
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
                if (databricks_permissions is not None and mlflow.get_tracking_uri() == "databricks"):
                    # Need to apply permissions
                    apply_model_permissions(reg_model_name=reg_model_name)

                model_src = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)

                tags = {
                    "start": message.get_meta(timestamp_column_name).min(),
                    "end": message.get_meta(timestamp_column_name).max(),
                    "count": message.get_meta(timestamp_column_name).count()
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

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node_full(MLFLOW_MODEL_WRITER, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
