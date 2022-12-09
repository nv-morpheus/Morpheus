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
import typing

import click
import yaml

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.logger import configure_logging
from stages.create_features import CreateFeaturesRWStage
from stages.preprocessing import PreprocessingRWStage


@click.command()
@click.option('--debug', default=False)
@click.option('--use_cpp', default=False)
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--n_dask_workers",
    default=6,
    type=click.IntRange(min=1),
    help="Number of dask workers",
)
@click.option(
    "--threads_per_dask_worker",
    default=2,
    type=click.IntRange(min=1),
    help="Number of threads per each dask worker",
)
@click.option(
    "--model_max_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--conf_file",
    type=click.STRING,
    default="./config/ransomware_detection.yaml",
    help="Ransomware detection configuration filepath",
)
@click.option(
    "--model_name",
    default="ransomw-model-short-rf",
    help="The name of the model that is deployed on Tritonserver",
)
@click.option("--server_url", required=True, help="Tritonserver url")
@click.option(
    "--sliding_window",
    default=3,
    type=click.IntRange(min=3),
    help="Sliding window to be used for model input request",
)
@click.option(
    '--input_glob',
    type=str,
    required=True,
    help=("Input glob pattern to match files to read. For example, './input_dir/*/snapshot-*/*.json' would read all "
          "files with the 'json' extension in the directory 'input_dir'."))
@click.option('--watch_directory',
              type=bool,
              default=False,
              help=("The watch directory option instructs this stage to not close down once all files have been read. "
                    "Instead it will read all files that match the 'input_glob' pattern, and then continue to watch "
                    "the directory for additional files. Any new files that are added that match the glob will then "
                    "be processed."))
@click.option(
    "--output_file",
    type=click.STRING,
    default="./ransomware_detection_output.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(debug,
                 use_cpp,
                 num_threads,
                 n_dask_workers,
                 threads_per_dask_worker,
                 model_max_batch_size,
                 conf_file,
                 model_name,
                 server_url,
                 sliding_window,
                 input_glob,
                 watch_directory,
                 output_file):

    if debug:
        configure_logging(log_level=logging.DEBUG)
    else:
        configure_logging(log_level=logging.INFO)

    snapshot_fea_length = 99

    CppConfig.set_should_use_cpp(use_cpp)

    # Its necessary to get the global config object and configure it for FIL mode
    config = Config()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = snapshot_fea_length * sliding_window
    config.class_labels = ["pred", "score"]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Load ransomware detection configuration
    rwd_conf = load_yaml(conf_file)

    # Exclude columns that are not required
    cols_exclude = ["SHA256"]

    # Only intrested plugins files will be read from Appshield snapshots
    interested_plugins = ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']

    # Columns from the above intrested plugins
    cols_interested_plugins = rwd_conf['raw_columns']

    # Feature columns used by the model
    feature_columns = rwd_conf['model_features']

    # File extensions
    file_extns = rwd_conf['file_extensions']

    # Set source stage
    # This stage reads raw data from the required plugins and merge all the plugins data into a single dataframe
    # for a given source.
    pipeline.set_source(
        AppShieldSourceStage(
            config,
            input_glob,
            interested_plugins,
            cols_interested_plugins,
            cols_exclude=cols_exclude,
            watch_directory=watch_directory,
        ))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="FromFile rate"))

    # Add create features stage
    # This stage generates model feature values from the raw data
    pipeline.add_stage(
        CreateFeaturesRWStage(config,
                              interested_plugins,
                              feature_columns,
                              file_extns,
                              n_workers=n_dask_workers,
                              threads_per_worker=threads_per_dask_worker))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="CreateFeatures rate"))

    # Add preprocessing stage.
    # This stage generates snapshot sequences using sliding window for each pid_process
    pipeline.add_stage(PreprocessingRWStage(config, feature_columns=feature_columns[:-1],
                                            sliding_window=sliding_window))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="PreProcessing rate"))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
        ))
    # # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Inference rate"))

    # Add a scores stage
    pipeline.add_stage(AddScoresStage(config, labels=["score"]))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="AddScore rate"))

    # Convert the probabilities to serialized JSON strings using the custom serialization stage
    pipeline.add_stage(SerializeStage(config, exclude=[r'^ID$', r'^_ts_', r'source_pid_process']))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="Serialize rate"))

    # Write the file to the output
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Add a monitor stage
    pipeline.add_stage(MonitorStage(config, description="ToFile rate"))

    # Run the pipeline
    pipeline.run()


def load_yaml(filepath: str) -> typing.Dict[object, object]:
    """
    This function loads yaml configuration to a dictionary

    Parameters
        ----------
        filepath : str
            A file's path

        Returns
        -------
        typing.Dict[object, object]
            Configuration as a dictionary
    """
    with open(filepath, 'r', encoding='utf8') as f:
        conf_dct = yaml.safe_load(f)
        f.close()

    return conf_dct


# Execution starts here
if __name__ == "__main__":
    run_pipeline()
