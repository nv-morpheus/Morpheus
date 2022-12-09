#!/usr/bin/env python
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

from morpheus.utils.decorators import register_module
from morpheus.utils.decorators import verify_module_registration


def test_register_module():

    @register_module("TestModule", "test_morpheus_modules")
    def module_init():
        return True

    assert module_init()

    # Attempting to register duplicate module raises an error.
    with pytest.raises(TypeError):

        @register_module(None, "test_morpheus_modules")
        def module_init2():
            pass

        module_init2()


def test_verify_module_registration():

    @verify_module_registration
    def verify_module_existence(module_id, namespace):
        return True

    assert verify_module_existence("TestModule", "test_morpheus_modules")

    with pytest.raises(TypeError):
        verify_module_existence(None, "test_morpheus_modules")

    with pytest.raises(Exception):
        verify_module_existence("TestModule2", "test_morpheus_modules")

    with pytest.raises(Exception):
        verify_module_existence("TestModule", "default")
