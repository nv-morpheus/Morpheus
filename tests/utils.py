# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import time
import typing

import cupy as cp
import mrc
import pandas as pd

import cudf

import morpheus
from morpheus._lib.common import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import df_to_csv
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.messages import ResponseMemoryProbs
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.inference import inference_stage
from morpheus.utils.atomic_integer import AtomicInteger


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
    Setting `columns` restricts the columns copied into probs to just the ones specified.
    Setting `order` specifies probs to be in either column or row major
    Setting `empty_probs` will create an empty probs array with 3 columns, and the same number of rows as the dataframe
    """

    def __init__(self,
                 c: Config,
                 expected_data_file: str = None,
                 columns: typing.List[str] = None,
                 order: str = 'K',
                 probs_type: str = 'f4',
                 empty_probs: bool = False):
        super().__init__(c)
        self._expected_data_file = expected_data_file
        self._columns = columns
        self._order = order
        self._probs_type = probs_type
        self._empty_probs = empty_probs

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
            if self._columns is not None:
                df = m.get_meta(self._columns)
            else:
                df = m.get_meta()

        if self._empty_probs:
            probs = cp.zeros([len(df), 3], 'float')
        else:
            probs = cp.array(df.values, dtype=self._probs_type, copy=True, order=self._order)

        memory = ResponseMemoryProbs(count=len(probs), probs=probs)

        return MultiResponseProbsMessage.from_message(m, memory=memory)

    def _build_single(self, builder: mrc.Builder, input_stream):
        stream = builder.make_node(self.unique_name, self._conv_message)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage


@register_stage("unittest-dfp-length-check")
class DFPLengthChecker(SinglePortStage):
    """
    Verifies that the incoming MessageMeta classes are of a specific length

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    expected_length : int
        Expected length of incoming messages

    num_exact: int
        Number of messages to check. For a datasource with 2000 records if we expect the first message to be of lenth
        1024, the next message will contain only 976. The first `num_exact` messages will be checked with `==` and `<=`
        after that.
    """

    def __init__(self, c: Config, expected_length: int, num_exact: int = 1):
        super().__init__(c)

        self._expected_length = expected_length
        self._num_exact = num_exact
        self._num_received = AtomicInteger(0)

    @property
    def name(self) -> str:
        return "dfp-length-check"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta)

    def supports_cpp_node(self):
        return False

    def _length_checker(self, x: MessageMeta):
        msg_num = self._num_received.get_and_inc()
        if msg_num < self._num_exact:
            assert x.count == self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} != {self._num_exact}"
        else:
            assert x.count <= self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} > {self._num_exact}"

        return x

    def _build_single(self, builder: mrc.Builder, input_stream):
        node = builder.make_node(self.unique_name, self._length_checker)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]


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


def get_column_names_from_file(file_name):
    df = read_file_to_df(file_name, file_type=FileTypes.Auto, df_type='pandas')
    return list(df.columns)


def extend_data(input_file, output_file, repeat_count):
    df = read_file_to_df(input_file, FileTypes.Auto, df_type='pandas')
    data = pd.concat([df for _ in range(repeat_count)])
    with open(output_file, 'w') as fh:
        output_strs = df_to_csv(data, include_header=True, include_index_col=False)
        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]
        fh.writelines(output_strs)


def assert_path_exists(filename: str, retry_count: int = 5, delay_ms: int = 500):
    """
    This should be used in place of `assert os.path.exists(filename)` inside of tests. This will automatically retry
    with a delay if the file is not immediately found. This removes the need for adding any `time.sleep()` inside of
    tests

    Parameters
    ----------
    filename : str
        The path to assert exists
    retry_count : int, optional
        Number of times to check for the file before failing, by default 5
    delay_ms : int, optional
        Milliseconds between trys, by default 500

    Returns
    -------
    Returns none but will throw an assertion error on failure.
    """

    # Quick exit if the file exists
    if (os.path.exists(filename)):
        return

    attempts = 1

    # Otherwise, delay and retry
    while (attempts <= retry_count):
        time.sleep(delay_ms / 1000.0)

        if (os.path.exists(filename)):
            return

        attempts += 1

    # Finally, actually assert on the final try
    assert os.path.exists(filename)


def duplicate_df_index(df: pd.DataFrame, replace_ids: typing.Dict[int, int]):

    # Return a new dataframe where we replace some index values with others
    return df.rename(index=replace_ids)


def duplicate_df_index_rand(df: pd.DataFrame, count=1):

    assert count * 2 <= len(df), "Count must be less than half the number of rows"

    # Sample 2x the count. One for the old ID and one for the new ID. Dont want duplicates so we use random.sample
    # (otherwise you could get less duplicates than requested if two IDs just swap)
    dup_ids = random.sample(df.index.values.tolist(), 2 * count)

    # Create a dictionary of old ID to new ID
    replace_dict = {x: y for x, y in zip(dup_ids[:count], dup_ids[count:])}

    # Return a new dataframe where we replace some index values with others
    return duplicate_df_index(df, replace_dict)


def create_df_with_dup_ids(tmp_path: str, dup_row=8) -> str:
    """
    Helper method to test issue #686, takes the filter_probs.csv and sets the id in row `dup_row` to the id of the
    previous row (or the next row if dup_row==0)
    """
    df = read_file_to_df(os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv'), file_type=FileTypes.Auto)
    assert df.index.is_unique

    data = df_to_csv(df, include_header=True, include_index_col=True, strip_newline=True)

    # Duplicate id=7
    dup_row_idx = dup_row + 1  # account for the header row
    if dup_row > 0:
        new_idx_val = dup_row - 1
    else:
        new_idx_val = 1

    data[dup_row_idx] = data[dup_row_idx].replace(str(dup_row), str(new_idx_val), 1)

    dup_file = os.path.join(tmp_path, 'dup_id.csv')
    with open(dup_file, 'w') as fh:
        fh.writelines("\n".join(data))

    return dup_file


def assert_df_equal(df_to_check: typing.Union[pd.DataFrame, cudf.DataFrame], val_to_check: typing.Any):

    # Comparisons work better in cudf so convert everything to that
    if (isinstance(df_to_check, cudf.DataFrame) or isinstance(df_to_check, cudf.Series)):
        df_to_check = df_to_check.to_pandas()

    if (isinstance(val_to_check, cudf.DataFrame) or isinstance(val_to_check, cudf.Series)):
        val_to_check = val_to_check.to_pandas()

    bool_df = df_to_check == val_to_check

    return bool(bool_df.all(axis=None))
