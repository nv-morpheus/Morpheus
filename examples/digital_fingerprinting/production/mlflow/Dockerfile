# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM --platform=$TARGETPLATFORM python:3.12-slim-bullseye

# Install curl for health check
RUN apt update && \
    apt install -y --no-install-recommends \
        curl libyaml-cpp-dev libyaml-dev && \
    apt autoremove -y && \
    apt clean all && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install python packages
RUN pip install "mlflow >=2.18,<3" boto3 pyyaml

# We run on port 5000
EXPOSE 5000

HEALTHCHECK CMD curl -f http://localhost:5000/health || exit 1
