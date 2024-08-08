/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/utilities/cudf_util.hpp"  // for CudfHelper

#include <pybind11/gil.h>

namespace morpheus::test {

class TestMessages : public morpheus::test::TestWithPythonInterpreter
{
  protected:
    void SetUp() override
    {
        morpheus::test::TestWithPythonInterpreter::SetUp();
        {
            pybind11::gil_scoped_acquire gil;

            // Initially I ran into an issue bootstrapping cudf, I was able to work-around the issue, details in:
            // https://github.com/rapidsai/cudf/issues/12862
            CudfHelper::load();
        }
    }
};

}  // namespace morpheus::test
