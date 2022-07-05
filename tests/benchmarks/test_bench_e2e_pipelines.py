# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import ConfigFIL
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.auto_encoder_inference_stage import AutoEncoderInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_ae_stage import PreprocessAEStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.stages.preprocess.train_ae_stage import TrainAEStage
from morpheus.utils.logging import configure_logging
from utils import TEST_DIRS

TEST_SOURCES = {
    "test_sid_nlp_e2e": {
        "file_path": os.path.join(TEST_DIRS.validation_data_dir, 'sid-validation-data.csv'),
        "repeat": 10
    },
     "test_abp_fil_e2e": {
        "file_path": os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines'),
        "repeat": 100
    },
     "test_phishing_nlp_e2e": {
        "file_path": os.path.join(TEST_DIRS.validation_data_dir, 'phishing-email-validation-data.jsonlines'),
        "repeat": 1
    },
     "test_cloudtrail_ae_e2e": {
        "glob_path": os.path.join(TEST_DIRS.validation_data_dir, 'hammah-*.csv'),
        "repeat": 1
    }
}

def nlp_pipeline(config: Config,
                     input_file,
                     repeat,
                     vocab_hash_file,
                     output_file,
                     model_name):

    configure_logging(log_level=logging.INFO)

    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PreprocessNLPStage(config,
                                          vocab_hash_file=vocab_hash_file,
                                          truncation=True,
                                          do_lower_case=True,
                                          add_special_tokens=False))
    pipeline.add_stage(TritonInferenceStage(config, model_name=model_name, server_url="localhost:8001", force_convert_inputs=True))
    pipeline.add_stage(AddClassificationsStage(config, threshold=0.5, prefix=""))
    pipeline.add_stage(MonitorStage(config))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.build()
    pipeline.run()

def fil_pipeline(config: Config,
                     input_file,
                     repeat,
                     output_file,
                     model_name):

    configure_logging(log_level=logging.INFO)

    pipeline = LinearPipeline(config)
    pipeline.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(PreprocessFILStage(config))
    pipeline.add_stage(TritonInferenceStage(config, model_name=model_name, server_url="localhost:8001", force_convert_inputs=True))
    pipeline.add_stage(AddClassificationsStage(config, threshold=0.5, prefix=""))
    pipeline.add_stage(MonitorStage(config))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.build()
    pipeline.run()

def ae_pipeline(config: Config,
                     input_glob,
                     repeat,
                     train_data_glob,
                     output_file):

    configure_logging(log_level=logging.INFO)
    pipeline = LinearPipeline(config)
    pipeline.set_source(CloudTrailSourceStage(config,
                                              input_glob=input_glob,
                                              max_files=200,
                                              repeat=repeat))
    pipeline.add_stage(TrainAEStage(config,
                                    train_data_glob=train_data_glob,
                                    seed=42))
    pipeline.add_stage(PreprocessAEStage(config))
    pipeline.add_stage(AutoEncoderInferenceStage(config))
    pipeline.add_stage(AddScoresStage(config))
    pipeline.add_stage(TimeSeriesStage(config,
                                       resolution="1m",
                                       min_window=" 12 h",
                                       hot_start=True,
                                       cold_end=False,
                                       filter_percent=90.0,
                                       zscore_threshold=8.0))
    pipeline.add_stage(MonitorStage(config))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    pipeline.build()
    pipeline.run()

def test_sid_nlp_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = 8
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 64
    config.feature_length = 256
    config.edge_buffer_size = 4
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    CppConfig.set_should_use_cpp(True)

    input_filepath = TEST_SOURCES["test_sid_nlp_e2e"]["file_path"]
    repeat = TEST_SOURCES["test_sid_nlp_e2e"]["repeat"]
    vocab_filepath = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    output_filepath = os.path.join(tmp_path, "sid_nlp_e2e_output.csv")
    model_name = "sid-minibert-onnx"

    benchmark(nlp_pipeline, config, input_filepath, repeat, vocab_filepath, output_filepath, model_name)

def test_abp_fil_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.FIL
    config.num_threads = 8
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 1024
    config.feature_length = 29
    config.edge_buffer_size = 4
    config.class_labels = ["mining"]
    config.fil = ConfigFIL()
    fil_cols_filepath = os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt')
    with open(fil_cols_filepath, "r") as lf:
        config.fil.feature_columns = [x.strip() for x in lf.readlines()]
    CppConfig.set_should_use_cpp(True)

    input_filepath = TEST_SOURCES["test_abp_fil_e2e"]["file_path"]
    repeat = TEST_SOURCES["test_abp_fil_e2e"]["repeat"]
    output_filepath = os.path.join(tmp_path, "abp_fil_e2e_output.csv")
    model_name = "abp-nvsmi-xgb"

    benchmark(fil_pipeline, config, input_filepath, repeat, output_filepath, model_name)


def test_phishing_nlp_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.NLP
    config.num_threads = 8
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 64
    config.feature_length = 128
    config.edge_buffer_size = 4
    config.class_labels = ["score", "pred"]
    CppConfig.set_should_use_cpp(True)

    input_filepath = TEST_SOURCES["test_phishing_nlp_e2e"]["file_path"]
    repeat = TEST_SOURCES["test_phishing_nlp_e2e"]["repeat"]
    vocab_filepath = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    output_filepath = os.path.join(tmp_path, "phishing_nlp_e2e_output.csv")
    model_name = "phishing-bert-onnx"

    benchmark(nlp_pipeline, config, input_filepath, repeat, vocab_filepath, output_filepath, model_name)


def test_cloudtrail_ae_e2e(benchmark, tmp_path):

    config = Config()
    config.mode = PipelineModes.AE
    config.num_threads = 1
    config.pipeline_batch_size = 1024
    config.model_max_batch_size = 1024
    config.feature_length = 32
    config.edge_buffer_size = 4
    config.class_labels = ["ae_anomaly_score"]

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentityaccountId"
    config.ae.userid_filter = "Account-123456789"
    ae_cols_filepath = os.path.join(TEST_DIRS.data_dir, 'columns_ae.txt')
    with open(ae_cols_filepath, "r") as lf:
        config.ae.feature_columns = [x.strip() for x in lf.readlines()]
    CppConfig.set_should_use_cpp(False)

    input_glob = TEST_SOURCES["test_cloudtrail_ae_e2e"]["glob_path"]
    print(input_glob)
    repeat = TEST_SOURCES["test_cloudtrail_ae_e2e"]["repeat"]
    train_data_glob = os.path.join(TEST_DIRS.training_data_dir, "hammah-*.csv")
    print(train_data_glob)
    output_filepath = os.path.join(tmp_path, "cloudtrail_ae_e2e_output.csv")

    benchmark(ae_pipeline, config, input_glob, repeat, train_data_glob, output_filepath)
