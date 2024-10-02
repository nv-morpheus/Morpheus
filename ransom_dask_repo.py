# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import sys
import typing

import click
import yaml

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.utils.logger import configure_logging

MORPHEUS_ROOT = os.environ["MORPHEUS_ROOT"]
ransom_example_dir = os.path.join(MORPHEUS_ROOT, "examples", "ransomware_detection")
sys.path.append(ransom_example_dir)
from cf import ModifiedCreateFeaturesRWStage

logger = logging.getLogger(f"morpheus.{__name__}")


@click.command()
@click.option('--use_triton', is_flag=True)
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--n_dask_workers",
    default=6,
    type=click.IntRange(min=1),
    help="Number of dask workers.",
)
@click.option(
    "--threads_per_dask_worker",
    default=2,
    type=click.IntRange(min=1),
    help="Number of threads per each dask worker.",
)
@click.option(
    "--model_max_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model.",
)
@click.option(
    "--conf_file",
    type=click.STRING,
    default=os.path.join(ransom_example_dir, "config/ransomware_detection.yaml"),
    help="Ransomware detection configuration filepath.",
)
@click.option(
    "--model_name",
    default="ransomw-model-short-rf",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--server_url", required=True, default="localhost:8000", help="Tritonserver url.")
@click.option(
    "--sliding_window",
    default=3,
    type=click.IntRange(min=3),
    help="Sliding window to be used for model input request.",
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
def run_pipeline(use_triton,
                 num_threads,
                 n_dask_workers,
                 threads_per_dask_worker,
                 model_max_batch_size,
                 conf_file,
                 model_name,
                 server_url,
                 sliding_window,
                 input_glob,
                 watch_directory):

    configure_logging(log_level=logging.DEBUG)

    snapshot_fea_length = 99

    # Its necessary to get the global config object and configure it for FIL mode.
    config = Config()
    config.mode = PipelineModes.FIL

    # Below properties are specified by the command line.
    config.num_threads = num_threads
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = snapshot_fea_length * sliding_window
    config.class_labels = ["pred", "score"]

    pipeline = LinearPipeline(config)
    rwd_conf = load_yaml(conf_file)

    cols_exclude = ["SHA256"]
    interested_plugins = ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']
    cols_interested_plugins = rwd_conf['raw_columns']
    model_features = rwd_conf['model_features']
    feature_columns = model_features + rwd_conf['features']
    file_extns = rwd_conf['file_extensions']

    pipeline.set_source(
        AppShieldSourceStage(
            config,
            input_glob,
            interested_plugins,
            cols_interested_plugins,
            cols_exclude=cols_exclude,
            watch_directory=watch_directory,
        ))

    pipeline.add_stage(MonitorStage(config, description="FromFile rate"))

    pipeline.add_stage(
        ModifiedCreateFeaturesRWStage(config,
                                      interested_plugins,
                                      feature_columns,
                                      file_extns,
                                      n_workers=n_dask_workers,
                                      threads_per_worker=threads_per_dask_worker))

    pipeline.add_stage(MonitorStage(config, description="CreateFeatures rate"))

    if use_triton:
        logger.info("Building TritonInferenceStage")
        pipeline.add_stage(
            TritonInferenceStage(
                config,
                model_name=model_name,
                server_url=server_url,
                force_convert_inputs=True,
            ))

    # Run the pipeline.
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
