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

import dfp.modules.dfp_data_prep  # noqa: F401
import dfp.modules.dfp_preproc  # noqa: F401
import dfp.modules.dfp_rolling_window  # noqa: F401
import dfp.modules.dfp_training  # noqa: F401
import mrc

import morpheus._lib.modules  # noqa: F401
import morpheus.modules.mlflow_model_writer  # noqa: F401
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import merge_dictionaries
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DATA_PREP
from ..utils.module_ids import DFP_PREPROC
from ..utils.module_ids import DFP_ROLLING_WINDOW
from ..utils.module_ids import DFP_TRAINING
from ..utils.module_ids import DFP_TRAINING_PIPE

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_TRAINING_PIPE, MORPHEUS_MODULE_NAMESPACE)
def dfp_training_pipe(builder: mrc.Builder):
    """
    This module function consolidates multiple dfp pipeline modules relevant to the training process into a single
    module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline builder instance.

    Notes
    -----
        Configurable parameters:
            - batching_options (dict): Options for batching the data; Example: See Below
            - cache_dir (str): Directory to cache the rolling window data; Example: "/path/to/cache/dir";
            Default: ./.cache
            - dfencoder_options (dict): Options for configuring the data frame encoder; Example: See Below
            - mlflow_writer_options (dict): Options for the MLflow model writer; Example: See Below
            - stream_aggregation_options (dict): Options for aggregating the data by stream; Example: See Below
            - timestamp_column_name (str): Name of the timestamp column used in the data; Example: "my_timestamp";
            Default: "timestamp"
            - user_splitting_options (dict): Options for splitting the data by user; Example: See Below

        batching_options:
            - end_time (datetime/string): Endtime of the time window; Example: "2023-03-14T23:59:59"; Default: None
            - iso_date_regex_pattern (string): Regex pattern for ISO date matching;
            Example: "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"; Default: <iso_date_regex_pattern>
            - parser_kwargs (dictionary): Additional arguments for the parser; Example: {}; Default: {}
            - period (string): Time period for grouping files; Example: "1d"; Default: "1d"
            - sampling_rate_s (integer): Sampling rate in seconds; Example: 60; Default: 60
            - start_time (datetime/string): Start time of the time window; Example: "2023-03-01T00:00:00"; Default: N

        dfencoder_options:
            - feature_columns (list): List of feature columns to train on; Example: ["column1", "column2", "column3"]
            - epochs (int): Number of epochs to train for; Example: 50
            - model_kwargs (dict): Keyword arguments to pass to the model; Example: {"encoder_layers": [64, 32],
            "decoder_layers": [32, 64], "activation": "relu", "swap_p": 0.1, "lr": 0.001, "lr_decay": 0.9,
            "batch_size": 32, "verbose": 1, "optimizer": "adam", "scalar": "min_max", "min_cats": 10,
            "progress_bar": false, "device": "cpu"}
            - validation_size (float): Size of the validation set; Example: 0.1

        mlflow_writer_options:
            - conda_env (string): Conda environment for the model; Example: `path/to/conda_env.yml`;
            Default: `[Required]`
            - databricks_permissions (dictionary): Permissions for the model; Example: See Below; Default: None
            - experiment_name_formatter (string): Formatter for the experiment name;
            Example: `experiment_name_{timestamp}`;
             Default: `[Required]`
            - model_name_formatter (string): Formatter for the model name; Example: `model_name_{timestamp}`;
            Default: `[Required]`
            - timestamp_column_name (string): Name of the timestamp column; Example: `timestamp`; Default: timestamp

        stream_aggregation_options:
            - cache_mode (string): The user ID to use if the user ID is not found; Example: 'batch'; Default: 'batch'
            - min_history (int): Minimum history to trigger a new training event; Example: 1; Default: 1
            - max_history (int): Maximum history to include in a new training event; Example: 0; Default: 0
            - timestamp_column_name (string): Name of the column containing timestamps; Example: 'timestamp';
            Default: 'timestamp'
            - aggregation_span (string): Lookback timespan for training data in a new training event; Example: '60d';
            Default: '60d'
            - cache_to_disk (bool): Whether or not to cache streaming data to disk; Example: false; Default: false
            - cache_dir (string): Directory to use for caching streaming data; Example: './.cache'; Default: './.cache'

        user_splitting_options:
            - fallback_username (str): The user ID to use if the user ID is not found; Example: "generic_user";
            Default: 'generic_user'
            - include_generic (bool): Whether to include a generic user ID in the output; Example: false; Default: False
            - include_individual (bool): Whether to include individual user IDs in the output; Example: true;
            Default: False
            - only_users (list): List of user IDs to include; others will be excluded;
            Example: ["user1", "user2", "user3"];
             Default: []
            - skip_users (list): List of user IDs to exclude from the output; Example: ["user4", "user5"]; Default: []
            - timestamp_column_name (str): Name of the column containing timestamps; Example: "timestamp";
            Default: 'timestamp'
            - userid_column_name (str): Name of the column containing user IDs; Example: "username"; Default: 'username'
    """

    config = builder.get_current_module_config()

    cache_dir = config.get("cache_dir")
    ts_column_name = config.get("timestamp_column_name")

    preproc_options = {
        "batching_options": config.get("batching_options", {}),
        "cache_dir": cache_dir,
        "pre_filter_options": {
            "enable_task_filtering": True, "filter_task_type": "training"
        },
        "timestamp_column_name": ts_column_name,
        "user_splitting_options": config.get("user_splitting_options", {}),
    }

    stream_aggregation_options = config.get("stream_aggregation_options", {})
    stream_aggregation_options = merge_dictionaries(stream_aggregation_options, {
        "cache_dir": cache_dir,
        "timestamp_column_name": ts_column_name,
    })

    data_prep_options = config.get("preprocessing_options", {})
    data_prep_options = merge_dictionaries(data_prep_options, {
        "timestamp_column_name": ts_column_name,
    })
    dfencoder_options = config.get("dfencoder_options", {})

    mlflow_writer_options = config.get("mlflow_writer_options", {
        "timestamp_column_name": ts_column_name,
    })

    preproc_defaults = {}
    preproc_conf = merge_dictionaries(preproc_options, preproc_defaults)

    stream_aggregation_defaults = {
        "cache_mode": "aggregate",
        "trigger_on_min_history": 300,
        "trigger_on_min_increment": 300,
    }
    dfp_rolling_window_conf = merge_dictionaries(stream_aggregation_options, stream_aggregation_defaults)

    data_prep_defaults = {}
    dfp_data_prep_conf = merge_dictionaries(data_prep_options, data_prep_defaults)

    dfp_training_defaults = {
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
    }
    dfp_training_conf = merge_dictionaries(dfencoder_options, dfp_training_defaults)

    mlflow_model_writer_defaults = {}
    mlflow_model_writer_conf = merge_dictionaries(mlflow_writer_options, mlflow_model_writer_defaults)

    # Load modules
    preproc_module = builder.load_module(DFP_PREPROC, "morpheus", "dfp_preproc", preproc_conf)
    dfp_rolling_window_module = builder.load_module(DFP_ROLLING_WINDOW,
                                                    "morpheus",
                                                    "dfp_rolling_window",
                                                    dfp_rolling_window_conf)
    dfp_data_prep_module = builder.load_module(DFP_DATA_PREP, "morpheus", "dfp_data_prep", dfp_data_prep_conf)
    dfp_training_module = builder.load_module(DFP_TRAINING, "morpheus", "dfp_training", dfp_training_conf)
    mlflow_model_writer_module = builder.load_module(MLFLOW_MODEL_WRITER,
                                                     "morpheus",
                                                     "mlflow_model_writer",
                                                     mlflow_model_writer_conf)

    # Make an edge between the modules.
    builder.make_edge(preproc_module.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_training_module.input_port("input"))
    builder.make_edge(dfp_training_module.output_port("output"), mlflow_model_writer_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", preproc_module.input_port("input"))
    builder.register_module_output("output", mlflow_model_writer_module.output_port("output"))
