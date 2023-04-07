/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_morpheus.hpp"  // IWYU pragma: associated
#include "morpheus/doca/doca_context.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include <rte_eal.h>
#include <doca_version.h>
#include <doca_argp.h>
#include <doca_gpunetio.h>
#include <doca_flow.h>

std::shared_ptr<morpheus::doca::doca_context> _context;

class TestDoca : public ::testing::Test
{
protected:
  void SetUp() override {
    // const std::lock_guard<std::mutex> lock(g_i_mutex);
    if (_context == nullptr) {
      _context = std::make_shared<morpheus::doca::doca_context>("a3:00.0", "a6:00.0");
    }
  }
};

TEST_F(TestDoca, SetupAndTeardown)
{
}
