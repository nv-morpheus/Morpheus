#!/bin/bash
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

# create a docker network for morpheus
docker network inspect morpheus >/dev/null 2>&1 || docker network create morpheus

# create the parent conda folder so it's found when mounting
mkdir -p .cache/conda/envs
mkdir -p ../.conda/pkgs

# create a config directory if it does not exist so it's found when mounting
mkdir -p ../.config
