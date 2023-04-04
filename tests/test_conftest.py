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

import pandas as pd
import pytest

import cudf

from dataset_loader import DatasetLoader
from morpheus.config import Config
from morpheus.config import CppConfig


@pytest.fixture(scope="function")
def cpp_from_marker(request: pytest.FixtureRequest) -> bool:

    use_cpp = len([x for x in request.node.iter_markers("use_cpp") if "added_by" in x.kwargs]) > 0
    use_python = len([x for x in request.node.iter_markers("use_python") if "added_by" in x.kwargs]) > 0

    assert use_cpp != use_python

    return use_cpp


@pytest.fixture(scope="function")
def df_type_from_marker(request: pytest.FixtureRequest) -> bool:

    use_cudf = len([x for x in request.node.iter_markers("use_cudf") if "added_by" in x.kwargs]) > 0
    use_pandas = len([x for x in request.node.iter_markers("use_pandas") if "added_by" in x.kwargs]) > 0

    assert use_cudf != use_pandas

    return "cudf" if use_cudf else "pandas"


@pytest.mark.use_cudf
@pytest.mark.use_pandas
def test_works_with_marks(dataset: DatasetLoader):
    # Test is parameterized so df runs twice, once as pandas and another time as cudf
    df = dataset["filter_probs.csv"]
    assert isinstance(df, (pd.DataFrame, cudf.DataFrame))


def test_only_pandas(dataset_pandas: DatasetLoader):
    # Test only runs with pandas
    df = dataset_pandas["filter_probs.csv"]
    assert isinstance(df, pd.DataFrame)


def test_only_cudf(dataset_cudf: DatasetLoader):
    # Test only runs with cudf
    df = dataset_cudf["filter_probs.csv"]
    assert isinstance(df, cudf.DataFrame)


def test_both(dataset: DatasetLoader):
    # By default, requesting dataset will parameterize both
    df = dataset["filter_probs.csv"]
    assert isinstance(df, (pd.DataFrame, cudf.DataFrame))


# === No Marks ===
def test_no_mark():
    assert CppConfig.get_should_use_cpp()


# === No Marks ===
@pytest.mark.use_cpp
def test_mark_use_cpp():
    assert CppConfig.get_should_use_cpp()


@pytest.mark.use_python
def test_mark_use_python():
    assert not CppConfig.get_should_use_cpp()


@pytest.mark.use_cpp
@pytest.mark.use_python
def test_mark_both(cpp_from_marker: bool):
    assert CppConfig.get_should_use_cpp() == cpp_from_marker


# === Marks and Config ===
@pytest.mark.use_cpp
def test_mark_and_config_use_cpp(config: Config):
    assert CppConfig.get_should_use_cpp()


@pytest.mark.use_python
def test_mark_and_config_use_python(config: Config):
    assert not CppConfig.get_should_use_cpp()


@pytest.mark.use_cpp
@pytest.mark.use_python
def test_mark_and_config_both(config: Config, cpp_from_marker: bool):
    assert CppConfig.get_should_use_cpp() == cpp_from_marker


def test_mark_and_config_neither(config: Config, cpp_from_marker: bool):
    assert CppConfig.get_should_use_cpp() == cpp_from_marker


# === Fixture ===
@pytest.mark.use_cpp
def test_fixture_use_cpp(use_cpp: bool):
    assert use_cpp
    assert CppConfig.get_should_use_cpp()


@pytest.mark.use_python
def test_fixture_use_python(use_cpp: bool):
    assert not use_cpp
    assert not CppConfig.get_should_use_cpp()


@pytest.mark.use_cpp
@pytest.mark.use_python
def test_fixture_both(use_cpp: bool):
    assert CppConfig.get_should_use_cpp() == use_cpp


def test_fixture_neither(use_cpp: bool):
    assert CppConfig.get_should_use_cpp() == use_cpp


# === Config Fixture ===
def test_config_fixture_no_cpp(config_no_cpp: Config):
    assert not CppConfig.get_should_use_cpp()


def test_config_fixture_only_cpp(config_only_cpp: Config):
    assert CppConfig.get_should_use_cpp()


class TestNoMarkerClass:

    def test_no_marker(self):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.use_python
    def test_python_marker(self):
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    def test_cpp_marker(self):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    @pytest.mark.use_python
    def test_marker_both(self, cpp_from_marker: bool):
        assert CppConfig.get_should_use_cpp() == cpp_from_marker

    @pytest.mark.slow
    def test_other_marker(self, use_cpp: bool):
        assert CppConfig.get_should_use_cpp() == use_cpp


@pytest.mark.use_python
class TestPythonMarkerClass:

    def test_no_marker(self):
        assert not CppConfig.get_should_use_cpp()

    def test_with_fixture(self, use_cpp: bool):
        assert not use_cpp
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.use_python
    def test_extra_marker(self):
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    def test_add_marker(self, cpp_from_marker: bool):
        assert CppConfig.get_should_use_cpp() == cpp_from_marker


@pytest.mark.use_cpp
class TestCppMarkerClass:

    def test_no_marker(self):
        assert CppConfig.get_should_use_cpp()

    def test_with_fixture(self, use_cpp: bool):
        assert use_cpp
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    def test_extra_marker(self):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.use_python
    def test_add_marker(self, cpp_from_marker: bool):
        assert CppConfig.get_should_use_cpp() == cpp_from_marker


# === DF Type ===
def test_df_type_no_marks(df_type, df_type_from_marker):
    assert df_type == df_type_from_marker


@pytest.mark.use_pandas
def test_df_type_pandas_marker(df_type):
    assert df_type == "pandas"


@pytest.mark.use_cudf
def test_df_type_cudf_marker(df_type):
    assert df_type == "cudf"


@pytest.mark.use_cudf
@pytest.mark.use_pandas
def test_df_type_both_markers(df_type, df_type_from_marker):
    assert df_type == df_type_from_marker
