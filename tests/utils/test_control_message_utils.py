# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import pytest

from morpheus.messages import ControlMessage
from morpheus.utils.control_message_utils import CMDefaultFailureContextManager
from morpheus.utils.control_message_utils import cm_set_failure
from morpheus.utils.control_message_utils import skip_processing_if_cm_failed


def test_cm_set_failure():
    cm = ControlMessage()
    reason = "Test Failure"

    assert cm_set_failure(cm, reason) == cm

    assert cm.get_metadata("cm_failed") is True
    assert cm.get_metadata("cm_failed_reason") == reason


def test_skip_forward_on_cm_failed():
    cm = ControlMessage()
    reason = "Test Failure"
    cm_set_failure(cm, reason)

    @skip_processing_if_cm_failed
    def dummy_func(cm, *args, **kwargs):
        return "Function Executed"

    assert dummy_func(cm) == cm

    cm2 = ControlMessage()
    assert dummy_func(cm2) == "Function Executed"


def test_CMDefaultFailureContextManager_no_exception():
    cm = ControlMessage()
    with CMDefaultFailureContextManager(cm) as c:
        pass
    with pytest.raises(RuntimeError):
        cm.get_metadata("cm_failed")


def test_CMDefaultFailureContextManager_with_exception():
    cm = ControlMessage()
    with CMDefaultFailureContextManager(cm) as c:
        raise Exception("Test Exception")

    assert cm.get_metadata("cm_failed") is True
    assert cm.get_metadata("cm_failed_reason") == "Test Exception"


if (__name__ == "__main__"):
    test_cm_set_failure()
    test_skip_forward_on_cm_failed()
    test_CMDefaultFailureContextManager_no_exception()
    test_CMDefaultFailureContextManager_with_exception()
