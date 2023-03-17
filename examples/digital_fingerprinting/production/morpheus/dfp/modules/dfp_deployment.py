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

import dfp.modules.dfp_inference_pipe  # noqa: F401
import dfp.modules.dfp_training_pipe  # noqa: F401
import mrc
from mrc.core.node import Broadcast

import morpheus.loaders.fsspec_loader  # noqa: F401
from morpheus.utils.loader_ids import FSSPEC_LOADER
from morpheus.utils.module_ids import DATA_LOADER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_utils import merge_dictionaries
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_DEPLOYMENT
from ..utils.module_ids import DFP_INFERENCE_PIPE
from ..utils.module_ids import DFP_TRAINING_PIPE

logger = logging.getLogger("morpheus.{}".format(__name__))


@register_module(DFP_DEPLOYMENT, MORPHEUS_MODULE_NAMESPACE)
def dfp_deployment(builder: mrc.Builder):
    """
    Parameters
    ----------
    builder : mrc.Builder
        Pipeline builder instance.

    Notes
    -----
    Configurable parameters:
        - training_options (dict): Options for the training pipeline module, including:
            - timestamp_column_name (str): Name of the timestamp column used in the data
            - cache_dir (str): Directory to cache the rolling window data
            - batching_options (dict): Options for batching the data, including:
                - end_time (datetime|str): End time of the time window
                - iso_date_regex_pattern (str): Regex pattern for ISO date matching
                - parser_kwargs (dict): Additional arguments for the parser
                - period (str): Time period for grouping files
                - sampling_rate_s (int): Sampling rate in seconds
                - start_time (datetime|str): Start time of the time window
            - user_splitting_options (dict): Options for splitting the data by user, including:
                - fallback_username (str): User ID to use if user ID not found (default: 'generic_user')
                - include_generic (bool): Include generic user ID in output (default: False)
                - include_individual (bool): Include individual user IDs in output (default: False)
                - only_users (list): List of user IDs to include in output, others will be excluded (default: [])
                - skip_users (list): List of user IDs to exclude from output (default: [])
                - timestamp_column_name (str): Name of column containing timestamps (default: 'timestamp')
                - userid_column_name (str): Name of column containing user IDs (default: 'username')
            - stream_aggregation_options (dict): Options for aggregating the data by stream
            - preprocessing_options (dict): Options for preprocessing the data
            - dfencoder_options (dict): Options for configuring the data frame encoder, used for training the model
            - mlflow_writer_options (dict): Options for the MLflow model writer, responsible for saving the trained model, including:
                - model_name_formatter (str): Format string for the model name, e.g. "model_{timestamp}"
                - experiment_name_formatter (str): Format string for the experiment name, e.g. "experiment_{timestamp}"
                - timestamp_column_name (str): Name of the timestamp column used in the data
                - conda_env (dict): Conda environment settings, including:
                    - channels (list): List of channels to use for the environment
                    - dependencies (list): List of dependencies for the environment
                    - pip (list): List of pip packages to install in the environment
                    - name (str): Name of the conda environment
        - inference_options (dict): Options for the inference pipeline module, including:
            - model_name_formatter (str): Format string for the model name, e.g. "model_{timestamp}"
            - fallback_username (str): User ID to use if user ID not found (default: 'generic_user')
            - timestamp_column_name (str): Name of the timestamp column in the input data
            - batching_options (dict): Options for batching the data, including:
                [omitted for brevity]
            - cache_dir (str): Directory to cache the rolling window data
            - detection_criteria (dict): Criteria for filtering detections, such as threshold and field_name
            - inference_options (dict): Options for the inference module, including model settings and other configurations
            - num_output_ports (int): Number of output ports for the module
            - preprocessing_options (dict): Options for preprocessing the data, including schema and timestamp column name
            - stream_aggregation_options (dict): Options for aggregating the data by stream, including:
                - aggregation_span (int): The time span for the aggregation window, in seconds
                - cache_to_disk (bool): Whether to cache the aggregated data to disk
            - user_splitting_options (dict): Options for splitting the data by user, including:
                [omitted for brevity]
            - write_to_file_options (dict): Options for writing the detections to a file, such as filename and overwrite settings
    """

    module_config = builder.get_current_module_config()

    num_output_ports = 2

    supported_loaders = {}
    fsspec_loader_defaults = {
        "loaders": [{
            "id": FSSPEC_LOADER
        }],
    }

    fsspec_dataloader_conf = merge_dictionaries(supported_loaders, fsspec_loader_defaults)

    dfp_training_pipe_conf = module_config["training_options"]
    dfp_inference_pipe_conf = module_config["inference_options"]

    fsspec_dataloader_module = builder.load_module(DATA_LOADER, "morpheus", "fsspec_dataloader", fsspec_dataloader_conf)
    dfp_training_pipe_module = builder.load_module(DFP_TRAINING_PIPE,
                                                   "morpheus",
                                                   "dfp_training_pipe",
                                                   dfp_training_pipe_conf)
    dfp_inference_pipe_module = builder.load_module(DFP_INFERENCE_PIPE,
                                                    "morpheus",
                                                    "dfp_inference_pipe",
                                                    dfp_inference_pipe_conf)

    # Create broadcast node to fork the pipeline.
    broadcast = Broadcast(builder, "broadcast")

    # Make an edge between modules
    builder.make_edge(fsspec_dataloader_module.output_port("output"), broadcast)
    builder.make_edge(broadcast, dfp_training_pipe_module.input_port("input"))
    builder.make_edge(broadcast, dfp_inference_pipe_module.input_port("input"))

    out_streams = [dfp_training_pipe_module.output_port("output"), dfp_inference_pipe_module.output_port("output")]

    # Register input port for a module.
    builder.register_module_input("input", fsspec_dataloader_module.input_port("input"))

    # Register output ports for a module.
    for i in range(num_output_ports):
        # Output ports are registered in increment order.
        builder.register_module_output(f"output-{i}", out_streams[i])
