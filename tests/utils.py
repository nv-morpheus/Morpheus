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

import morpheus
from morpheus._lib.file_types import FileTypes
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


class ConvMsg(SinglePortStage):
    """
    Simple test stage to convert a MultiMessage to a MultiResponseProbsMessage
    Basically a cheap replacement for running an inference stage.

    Setting `expected_data_file` to the path of a cav/json file will cause the probs array to be read from file.
    Setting `expected_data_file` to `None` causes the probs array to be a copy of the incoming dataframe.
    """

    def __init__(self, c: Config, expected_data_file: str = None):
        super().__init__(c)
        self._expected_data_file = expected_data_file

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
            df = m.meta.df

        probs = df.values
        memory = ResponseMemoryProbs(count=len(probs), probs=probs)
        return MultiResponseProbsMessage(m.meta, 0, len(probs), memory, 0, len(probs))

    def _build_single(self, seg, input_stream):
        stream = seg.make_node(self.unique_name, self._conv_message)
        seg.make_edge(input_stream[0], stream)

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
