/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

uint32_t const PACKETS_PER_THREAD   = 4;
uint32_t const THREADS_PER_BLOCK    = 512;
uint32_t const PACKETS_PER_BLOCK    = PACKETS_PER_THREAD * THREADS_PER_BLOCK;
uint32_t const PACKET_RX_TIMEOUT_NS = 5000;

uint32_t const MAX_PKT_RECEIVE = PACKETS_PER_BLOCK;
uint32_t const MAX_PKT_SIZE    = 8192;
uint32_t const MAX_PKT_NUM     = 65536;
