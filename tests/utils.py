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
import pandas as pd

import cudf

import morpheus
from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import df_to_csv
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


def extend_df(df, repeat_count) -> pd.DataFrame:
    extended_df = pd.concat([df for _ in range(repeat_count)])
    return extended_df.reset_index(inplace=False, drop=True)


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


def assert_df_equal(df_to_check: typing.Union[pd.DataFrame, cudf.DataFrame], val_to_check: typing.Any):

    # Comparisons work better in cudf so convert everything to that
    if (isinstance(df_to_check, cudf.DataFrame) or isinstance(df_to_check, cudf.Series)):
        df_to_check = df_to_check.to_pandas()

    if (isinstance(val_to_check, cudf.DataFrame) or isinstance(val_to_check, cudf.Series)):
        val_to_check = val_to_check.to_pandas()
    elif (isinstance(val_to_check, cp.ndarray)):
        val_to_check = val_to_check.get()

    bool_df = df_to_check == val_to_check

    return bool(bool_df.all(axis=None))


def assert_results(results: dict) -> dict:
    """
    Receives the results dict from the `CompareDataframeStage.get_results` method,
    and asserts that all columns and rows match
    """
    assert results["diff_cols"] == 0, f"Expected diff_cols=0 : {results}"
    assert results["diff_rows"] == 0, f"Expected diff_rows=0 : {results}"
    return results
