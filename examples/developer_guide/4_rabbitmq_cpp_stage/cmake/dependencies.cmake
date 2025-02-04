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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

rapids_find_package(CUDAToolkit REQUIRED)
rapids_find_package(cudf REQUIRED)

set(RABBITMQ_VERSION "0.12.0" CACHE STRING "Version of RabbitMQ-C to use")
include(Configure_rabbitmq)

set(SIMPLE_AMQP_CLIENT_VERSION "2.5.1" CACHE STRING "Version of SimpleAmqpClient to use")
include(Configure_SimpleAmqpClient)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
