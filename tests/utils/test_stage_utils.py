# Copyright (c) 2024, NVIDIA CORPORATION.
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

import typing

import mrc
import pytest
from mrc import Subscriber

from morpheus.utils.stage_utils import fn_receives_subscriber


@pytest.mark.parametrize("has_arg, type_hint, expected",
                         [(False, None, False), (True, float, False), (True, "int", False),
                          (True, mrc.Subscriber, True), (True, Subscriber, True), (True, "mrc.Subscriber", True),
                          (True, "Subscriber", True)])
def test_fn_receives_subscriber(has_arg: bool, type_hint: typing.Any, expected: bool):
    if has_arg:

        def test_fn(first_arg: type_hint):
            pass
    else:

        def test_fn():
            pass

    assert fn_receives_subscriber(test_fn) is expected
