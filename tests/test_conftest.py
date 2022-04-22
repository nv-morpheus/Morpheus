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

import pytest

from morpheus.config import Config
from morpheus.config import CppConfig


@pytest.mark.use_python
def test_mark_no_cpp(config: Config):
    assert not CppConfig.get_should_use_cpp(), "Incorrect use_cpp"


@pytest.mark.use_cpp
def test_mark_only_cpp(config: Config):
    assert CppConfig.get_should_use_cpp(), "Incorrect use_cpp"


def test_mark_neither(config: Config):
    pass


def test_explicit_fixture_no_cpp(config_no_cpp: Config):
    assert not CppConfig.get_should_use_cpp(), "Incorrect use_cpp"


def test_explicit_fixture_only_cpp(config_only_cpp: Config):
    assert CppConfig.get_should_use_cpp(), "Incorrect use_cpp"


class TestNoMarkerClass:

    def test_no_marker(self, config: Config):
        pass

    @pytest.mark.use_python
    def test_python_marker(self, config: Config):
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    def test_cpp_marker(self, config: Config):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.slow
    def test_other_marker(self, config: Config):
        pass


@pytest.mark.use_python
class TestPythonMarkerClass:

    def test_no_marker(self, config: Config):
        assert not CppConfig.get_should_use_cpp()

    @pytest.mark.use_python
    def test_extra_marker(self, config: Config):
        assert not CppConfig.get_should_use_cpp()


@pytest.mark.use_cpp
class TestCppMarkerClass:

    def test_no_marker(self, config: Config):
        assert CppConfig.get_should_use_cpp()

    @pytest.mark.use_cpp
    def test_extra_marker(self, config: Config):
        assert CppConfig.get_should_use_cpp()
