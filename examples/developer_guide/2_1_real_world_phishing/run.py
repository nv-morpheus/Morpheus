#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from recipient_features_stage import RecipientFeaturesStage

import morpheus
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging


def run_pipeline():
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    triton_url = os.environ.get('TRITON_URL', 'localhost:8001')
    root_dir = os.environ['MORPHEUS_ROOT']
    out_dir = os.environ.get('OUT_DIR', '/tmp')

    labels_file = os.path.join(morpheus.DATA_DIR, 'labels_phishing.txt')
    vocab_file = os.path.join(morpheus.DATA_DIR, 'bert-base-uncased-hash.txt')

    input_file = os.path.join(root_dir, 'examples/data/email_with_addresses.jsonlines')
    results_file = os.path.join(out_dir, 'detections.jsonlines')

    # It's necessary to configure the pipeline for NLP mode
    config = Config()
    config.mode = PipelineModes.NLP

    # Set the thread count to match our cpu count
    config.num_threads = os.cpu_count()
    config.feature_length = 128

    with open(labels_file) as fh:
        config.class_labels = [x.strip() for x in fh]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add our custom stage
    pipeline.add_stage(RecipientFeaturesStage(config))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Tokenize the input
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name='phishing-bert-onnx',
            server_url=triton_url,
            force_convert_inputs=True,
        ))

    # Monitor the inference rate
    pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))

    # Filter values lower than 0.9
    pipeline.add_stage(FilterDetectionsStage(config, threshold=0.9))

    # Write the to the output file
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=results_file, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
