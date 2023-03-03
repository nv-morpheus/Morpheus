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

import os

from dfp.utils.dfp_arg_parser import DFPArgParser
from dfp.utils.dfp_arg_parser import pyobj2str
from dfp.utils.module_ids import DFP_DATA_PREP
from dfp.utils.module_ids import DFP_DEPLOYMENT
from dfp.utils.module_ids import DFP_INFERENCE
from dfp.utils.module_ids import DFP_INFERENCE_PIPE
from dfp.utils.module_ids import DFP_POST_PROCESSING
from dfp.utils.module_ids import DFP_PREPROC
from dfp.utils.module_ids import DFP_ROLLING_WINDOW
from dfp.utils.module_ids import DFP_SPLIT_USERS
from dfp.utils.module_ids import DFP_TRAINING
from dfp.utils.module_ids import DFP_TRAINING_PIPE
from dfp.utils.regex_utils import iso_date_regex_pattern
from dfp.utils.schema_utils import Schema

from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.messages.multi_message import MultiMessage
from morpheus.utils.loader_ids import FILE_TO_DF_LOADER
from morpheus.utils.loader_ids import FSSPEC_LOADER
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_ids import WRITE_TO_FILE


class ConfigGenerator:

    def __init__(self, config: Config, dfp_arg_parser: DFPArgParser, schema: Schema, encoding: str = "latin1"):
        self._config = config
        self._dfp_arg_parser = dfp_arg_parser
        self._encoding = encoding
        self._source_schema_str = pyobj2str(schema.source, encoding=encoding)
        self._preprocess_schema_str = pyobj2str(schema.preprocess, encoding=encoding)
        self._input_message_type = pyobj2str(MultiMessage, encoding)

    def get_module_conf(self):
        module_conf = {}

        module_conf["module_id"] = DFP_DEPLOYMENT
        module_conf["module_name"] = "dfp_deployment"
        module_conf["namespace"] = MODULE_NAMESPACE

        module_conf[FSSPEC_LOADER] = self.fsspec_dataloader_module_conf()
        module_conf[DFP_TRAINING_PIPE] = self.train_module_conf()
        module_conf[DFP_INFERENCE_PIPE] = self.infer_module_conf()
        module_conf["output_port_count"] = 2

        return module_conf

    def fsspec_dataloader_module_conf(self):
        module_conf = {"loaders": [{"id": FSSPEC_LOADER}]}
        return module_conf

    def infer_module_conf(self):
        module_conf = {
            DFP_PREPROC: {
                FILE_BATCHER: {
                    "period": "D",
                    "sampling_rate_s": self._dfp_arg_parser.sample_rate_s,
                    "start_time": self._dfp_arg_parser.time_fields.start_time,
                    "end_time": self._dfp_arg_parser.time_fields.end_time,
                    "iso_date_regex_pattern": iso_date_regex_pattern,
                    "timestamp_column_name": self._config.ae.timestamp_column_name,
                    "parser_kwargs": {
                        "lines": False, "orient": "records"
                    },
                    "cache_dir": self._dfp_arg_parser.cache_dir,
                    "filter_null": True,
                    "file_type": "JSON",
                    "schema": {
                        "schema_str": self._source_schema_str, "encoding": self._encoding
                    },
                    "task_type": "inference"
                },
                FILE_TO_DF_LOADER: {
                    "loaders": [{
                        "id": FILE_TO_DF_LOADER
                    }], "module_name": "dfp_file_to_df_dataloader_inf"
                },
                DFP_SPLIT_USERS: {
                    "include_generic": self._dfp_arg_parser.include_generic,
                    "include_individual": self._dfp_arg_parser.include_individual,
                    "skip_users": self._dfp_arg_parser.skip_users,
                    "only_users": self._dfp_arg_parser.only_users,
                    "timestamp_column_name": self._config.ae.timestamp_column_name,
                    "userid_column_name": self._config.ae.userid_column_name,
                    "fallback_username": self._config.ae.fallback_username
                }
            },
            DFP_ROLLING_WINDOW: {
                "min_history": 1,
                "min_increment": 0,
                "max_history": "1d",
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
            },
            DFP_DATA_PREP: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                },
            },
            DFP_INFERENCE: {
                "model_name_formatter": self._dfp_arg_parser.model_name_formatter,
                "fallback_username": self._config.ae.fallback_username,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            FILTER_DETECTIONS: {
                "field_name": "mean_abs_z",
                "threshold": 2.0,
                "filter_source": "DATAFRAME",
                "schema": {
                    "input_message_type": self._input_message_type, "encoding": self._encoding
                }
            },
            DFP_POST_PROCESSING: {
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            SERIALIZE: {
                "exclude": ['batch_count', 'origin_hash', '_row_hash', '_batch_id'],
                "use_cpp": CppConfig.get_should_use_cpp()
            },
            WRITE_TO_FILE: {
                "filename": "dfp_detections_{}.csv".format(self._dfp_arg_parser.source), "overwrite": True
            }
        }

        return module_conf

    def train_module_conf(self):
        module_conf = {
            DFP_PREPROC: {
                FILE_BATCHER: {
                    "period": "D",
                    "sampling_rate_s": self._dfp_arg_parser.sample_rate_s,
                    "start_time": self._dfp_arg_parser.time_fields.start_time,
                    "end_time": self._dfp_arg_parser.time_fields.end_time,
                    "iso_date_regex_pattern": iso_date_regex_pattern,
                    "timestamp_column_name": self._config.ae.timestamp_column_name,
                    "parser_kwargs": {
                        "lines": False, "orient": "records"
                    },
                    "cache_dir": self._dfp_arg_parser.cache_dir,
                    "filter_null": True,
                    "file_type": "JSON",
                    "schema": {
                        "schema_str": self._source_schema_str, "encoding": self._encoding
                    },
                    "task_type": "training"
                },
                FILE_TO_DF_LOADER: {
                    "loaders": [{
                        "id": FILE_TO_DF_LOADER
                    }], "module_name": "dfp_file_to_df_dataloader_tra"
                },
                DFP_SPLIT_USERS: {
                    "include_generic": self._dfp_arg_parser.include_generic,
                    "include_individual": self._dfp_arg_parser.include_individual,
                    "skip_users": self._dfp_arg_parser.skip_users,
                    "only_users": self._dfp_arg_parser.only_users,
                    "timestamp_column_name": self._config.ae.timestamp_column_name,
                    "userid_column_name": self._config.ae.userid_column_name,
                    "fallback_username": self._config.ae.fallback_username
                }
            },
            DFP_ROLLING_WINDOW: {
                "min_history": 300,
                "min_increment": 300,
                "max_history": self._dfp_arg_parser.duration,
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            DFP_DATA_PREP: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                }
            },
            DFP_TRAINING: {
                "model_kwargs": {
                    "encoder_layers": [512, 500],  # layers of the encoding part
                    "decoder_layers": [512],  # layers of the decoding part
                    "activation": 'relu',  # activation function
                    "swap_p": 0.2,  # noise parameter
                    "lr": 0.001,  # learning rate
                    "lr_decay": 0.99,  # learning decay
                    "batch_size": 512,
                    "verbose": False,
                    "optimizer": 'sgd',  # SGD optimizer is selected(Stochastic gradient descent)
                    "scaler": 'standard',  # feature scaling method
                    "min_cats": 1,  # cut off for minority categories
                    "progress_bar": False,
                    "device": "cuda"
                },
                "feature_columns": self._config.ae.feature_columns,
                "epochs": 30,
                "validation_size": 0.10
            },
            MLFLOW_MODEL_WRITER: {
                "model_name_formatter": self._dfp_arg_parser.model_name_formatter,
                "experiment_name_formatter": self._dfp_arg_parser.experiment_name_formatter,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "conda_env": {
                    'channels': ['defaults', 'conda-forge'],
                    'dependencies': ['python={}'.format('3.8'), 'pip'],
                    'pip': ['mlflow', 'dfencoder'],
                    'name': 'mlflow-env'
                },
                "databricks_permissions": None
            }
        }

        return module_conf

    def inf_pipe_module_conf(self):
        module_conf = {
            FILE_BATCHER: {
                "period": "D",
                "sampling_rate_s": self._dfp_arg_parser.sample_rate_s,
                "start_time": self._dfp_arg_parser.time_fields.start_time,
                "end_time": self._dfp_arg_parser.time_fields.end_time,
                "iso_date_regex_pattern": iso_date_regex_pattern
            },
            FILE_TO_DF: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "parser_kwargs": {
                    "lines": False, "orient": "records"
                },
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "filter_null": True,
                "file_type": "JSON",
                "schema": {
                    "schema_str": self._source_schema_str, "encoding": self._encoding
                }
            },
            DFP_SPLIT_USERS: {
                "include_generic": self._dfp_arg_parser.include_generic,
                "include_individual": self._dfp_arg_parser.include_individual,
                "skip_users": self._dfp_arg_parser.skip_users,
                "only_users": self._dfp_arg_parser.only_users,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "userid_column_name": self._config.ae.userid_column_name,
                "fallback_username": self._config.ae.fallback_username
            },
            DFP_ROLLING_WINDOW: {
                "min_history": 1,
                "min_increment": 0,
                "max_history": "1d",
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            DFP_DATA_PREP: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                }
            },
            DFP_INFERENCE: {
                "model_name_formatter": self._dfp_arg_parser.model_name_formatter,
                "fallback_username": self._config.ae.fallback_username,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            FILTER_DETECTIONS: {
                "field_name": "mean_abs_z",
                "threshold": 2.0,
                "filter_source": "DATAFRAME",
                "schema": {
                    "input_message_type": self._input_message_type, "encoding": self._encoding
                }
            },
            DFP_POST_PROCESSING: {
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            SERIALIZE: {
                "exclude": ['batch_count', 'origin_hash', '_row_hash', '_batch_id']
            },
            WRITE_TO_FILE: {
                "filename": "dfp_detections_{}.csv".format(self._dfp_arg_parser.source), "overwrite": True
            }
        }

        return module_conf

    def tra_pipe_module_conf(self):
        module_conf = {
            FILE_BATCHER: {
                "period": "D",
                "sampling_rate_s": self._dfp_arg_parser.sample_rate_s,
                "start_time": self._dfp_arg_parser.time_fields.start_time,
                "end_time": self._dfp_arg_parser.time_fields.end_time,
                "iso_date_regex_pattern": iso_date_regex_pattern
            },
            FILE_TO_DF: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "parser_kwargs": {
                    "lines": False, "orient": "records"
                },
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "filter_null": True,
                "file_type": "JSON",
                "schema": {
                    "schema_str": self._source_schema_str, "encoding": self._encoding
                }
            },
            DFP_SPLIT_USERS: {
                "include_generic": self._dfp_arg_parser.include_generic,
                "include_individual": self._dfp_arg_parser.include_individual,
                "skip_users": self._dfp_arg_parser.skip_users,
                "only_users": self._dfp_arg_parser.only_users,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "userid_column_name": self._config.ae.userid_column_name,
                "fallback_username": self._config.ae.fallback_username
            },
            DFP_ROLLING_WINDOW: {
                "min_history": 300,
                "min_increment": 300,
                "max_history": self._dfp_arg_parser.duration,
                "cache_dir": self._dfp_arg_parser.cache_dir,
                "timestamp_column_name": self._config.ae.timestamp_column_name
            },
            DFP_DATA_PREP: {
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "schema": {
                    "schema_str": self._preprocess_schema_str, "encoding": self._encoding
                }
            },
            DFP_TRAINING: {
                "model_kwargs": {
                    "encoder_layers": [512, 500],  # layers of the encoding part
                    "decoder_layers": [512],  # layers of the decoding part
                    "activation": 'relu',  # activation function
                    "swap_p": 0.2,  # noise parameter
                    "lr": 0.001,  # learning rate
                    "lr_decay": 0.99,  # learning decay
                    "batch_size": 512,
                    "verbose": False,
                    "optimizer": 'sgd',  # SGD optimizer is selected(Stochastic gradient descent)
                    "scaler": 'standard',  # feature scaling method
                    "min_cats": 1,  # cut off for minority categories
                    "progress_bar": False,
                    "device": "cuda"
                },
                "feature_columns": self._config.ae.feature_columns,
                "epochs": 30,
                "validation_size": 0.10
            },
            MLFLOW_MODEL_WRITER: {
                "model_name_formatter": self._dfp_arg_parser.model_name_formatter,
                "experiment_name_formatter": self._dfp_arg_parser.experiment_name_formatter,
                "timestamp_column_name": self._config.ae.timestamp_column_name,
                "conda_env": {
                    'channels': ['defaults', 'conda-forge'],
                    'dependencies': ['python={}'.format('3.8'), 'pip'],
                    'pip': ['mlflow', 'dfencoder'],
                    'name': 'mlflow-env'
                },
                "databricks_permissions": None
            }
        }

        return module_conf


def generate_ae_config(source: str,
                       userid_column_name: str,
                       timestamp_column_name: str,
                       pipeline_batch_size: int = 0,
                       edge_buffer_size: int = 0,
                       use_cpp: bool = False,
                       num_threads: int = os.cpu_count()):
    config = Config()

    CppConfig.set_should_use_cpp(use_cpp)

    config.num_threads = num_threads

    if pipeline_batch_size > 0:
        config.pipeline_batch_size = pipeline_batch_size

    if edge_buffer_size > 0:
        config.edge_buffer_size = edge_buffer_size

    config.ae = ConfigAutoEncoder()

    labels_file = "data/columns_ae_{}.txt".format(source)
    config.ae.feature_columns = load_labels_file(get_package_relative_file(labels_file))
    config.ae.userid_column_name = userid_column_name
    config.ae.timestamp_column_name = timestamp_column_name

    return config
