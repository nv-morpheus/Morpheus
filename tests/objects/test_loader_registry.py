#!/usr/bin/env python
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

import mrc
import cudf

import morpheus.messages as messages
import morpheus._lib.messages as _messages


def test_loader_registry():
    def csv_test_loader(control_message: messages.MessageControl):
        config = control_message.config()
        if ('files' not in config):
            raise ValueError("No files specified in config")
        files = config['files']

        df = None
        for file in files:
            filepath = file['path']
            if df is None:
                df = cudf.read_csv(filepath)
            else:
                df = df.append(cudf.read_csv(filepath))

        return messages.MessageMeta(df)

    _messages.DataLoaderRegistry.register_loader("csv", csv_test_loader)