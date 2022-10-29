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

import collections
import json
import os
import time

import cupy as cp
import srf

import morpheus
from morpheus._lib.file_types import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.messages import ResponseMemoryProbs
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.inference import inference_stage


class TestDirectories(object):

    def __init__(self, cur_file=__file__) -> None:
        self.tests_dir = os.path.dirname(cur_file)
        self.morpheus_root = os.environ.get('MORPHEUS_ROOT', os.path.dirname(self.tests_dir))
        self.data_dir = morpheus.DATA_DIR
        self.models_dir = os.path.join(self.morpheus_root, 'models')
        self.datasets_dir = os.path.join(self.models_dir, 'datasets')
        self.training_data_dir = os.path.join(self.datasets_dir, 'training-data')
        self.validation_data_dir = os.path.join(self.datasets_dir, 'validation-data')
        self.tests_data_dir = os.path.join(self.tests_dir, 'tests_data')
        self.mock_triton_servers_dir = os.path.join(self.tests_dir, 'mock_triton_server')


TEST_DIRS = TestDirectories()


@register_stage("unittest-conv-msg")
class ConvMsg(SinglePortStage):
    """
    Simple test stage to convert a MultiMessage to a MultiResponseProbsMessage
    Basically a cheap replacement for running an inference stage.

    Setting `expected_data_file` to the path of a cav/json file will cause the probs array to be read from file.
    Setting `expected_data_file` to `None` causes the probs array to be a copy of the incoming dataframe.
    Setting `order` specifies probs to be in either column or row major
    """

    def __init__(self, c: Config, expected_data_file: str = None, order: str = 'K'):
        super().__init__(c)
        self._expected_data_file = expected_data_file
        self._order = order

    @property
    def name(self):
        return "test"

    def accepted_types(self):
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _conv_message(self, m):
        if self._expected_data_file is not None:
            df = read_file_to_df(self._expected_data_file, FileTypes.CSV, df_type="cudf")
        else:
            df = m.get_meta()

        probs = cp.array(df.values, copy=True, order=self._order)
        memory = ResponseMemoryProbs(count=len(probs), probs=probs)
        return MultiResponseProbsMessage(m.meta, 0, len(probs), memory, 0, len(probs))

    def _build_single(self, builder: srf.Builder, input_stream):
        stream = builder.make_node(self.unique_name, self._conv_message)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage


class IW(inference_stage.InferenceWorker):
    """
    Concrete impl class of `InferenceWorker` for the purposes of testing
    """

    def calc_output_dims(self, _):
        # Intentionally calling the abc empty method for coverage
        super().calc_output_dims(_)
        return (1, 2)


Results = collections.namedtuple('Results', ['total_rows', 'diff_rows', 'error_pct'])


def calc_error_val(results_file):
    """
    Based on the calc_error_val function in val-utils.sh
    """
    with open(results_file) as fh:
        results = json.load(fh)

    total_rows = results['total_rows']
    diff_rows = results['diff_rows']
    return Results(total_rows=total_rows, diff_rows=diff_rows, error_pct=(diff_rows / total_rows) * 100)


def write_file_to_kafka(bootstrap_servers: str,
                        kafka_topic: str,
                        input_file: str,
                        client_id: str = 'morpheus_unittest_writer') -> int:
    """
    Writes data from `inpute_file` into a given Kafka topic, emitting one message for each line int he file.
    Returning the number of messages written
    """
    from kafka import KafkaProducer
    num_records = 0
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers, client_id=client_id)
    with open(input_file) as fh:
        for line in fh:
            producer.send(kafka_topic, line.strip().encode('utf-8'))
            num_records += 1

    producer.flush()

    assert num_records > 0
    time.sleep(1)

    return num_records


def compare_class_to_scores(file_name, field_names, class_prefix, score_prefix, threshold):
    df = read_file_to_df(file_name, file_type=FileTypes.Auto, df_type='pandas')
    for field_name in field_names:
        class_field = f"{class_prefix}{field_name}"
        score_field = f"{score_prefix}{field_name}"
        above_thresh = df[score_field] > threshold

        df[class_field].to_csv(f"/tmp/class_field_{field_name}.csv")
        df[score_field].to_csv(f"/tmp/score_field_vals_{field_name}.csv")
        above_thresh.to_csv(f"/tmp/score_field_{field_name}.csv")

        assert all(above_thresh == df[class_field]), f"Mismatch on {field_name}"
