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
    This module function allows for the consolidation of multiple dfp pipeline modules relevent to training
    process into a single module.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.

    Notes
    ----------
    Configurable parameters:
        - timestamp_column_name : str
        - cache_dir : str
        - batching_options : dict
        - user_splitting_options : dict
        - stream_aggregation_options : dict
        - preprocessing_options : dict
        - dfencoder_options : dict
        - mlflow_writer_options : dict
    """

    config = builder.get_current_module_config()

    cache_dir = config.get("cache_dir")
    ts_column_name = config.get("timestamp_column_name")

    preproc_options = {
        "batching_options": config.get("batching_options", {}),
        "cache_dir": cache_dir,
        "pre_filter_options": {
            "enable_task_filtering": True,
            "filter_task_type": "training"
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

    # TODO(Devin): Not sure, but it seems like this is the right place to be opinionated about these values
    # mostly because dfencoder itself has default values so we don't need them at the dfp_training level
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
    dfp_rolling_window_module = builder.load_module(DFP_ROLLING_WINDOW, "morpheus", "dfp_rolling_window",
                                                    dfp_rolling_window_conf)
    dfp_data_prep_module = builder.load_module(DFP_DATA_PREP, "morpheus", "dfp_data_prep", dfp_data_prep_conf)
    dfp_training_module = builder.load_module(DFP_TRAINING, "morpheus", "dfp_training", dfp_training_conf)
    mlflow_model_writer_module = builder.load_module(MLFLOW_MODEL_WRITER, "morpheus", "mlflow_model_writer",
                                                     mlflow_model_writer_conf)

    # Make an edge between the modules.
    builder.make_edge(preproc_module.output_port("output"), dfp_rolling_window_module.input_port("input"))
    builder.make_edge(dfp_rolling_window_module.output_port("output"), dfp_data_prep_module.input_port("input"))
    builder.make_edge(dfp_data_prep_module.output_port("output"), dfp_training_module.input_port("input"))
    builder.make_edge(dfp_training_module.output_port("output"), mlflow_model_writer_module.input_port("input"))

    # Register input and output port for a module.
    builder.register_module_input("input", preproc_module.input_port("input"))
    builder.register_module_output("output", mlflow_model_writer_module.output_port("output"))
