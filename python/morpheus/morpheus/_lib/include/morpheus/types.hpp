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

#include <cudf/types.hpp>

#include <cstddef>  // for size_t
#include <map>
#include <string>
#include <utility>  // for pair
#include <vector>

namespace mrc::coroutines {
template <typename T>
class Task;
}

namespace morpheus {

struct TensorObject;

/**
 * @addtogroup objects
 * @{
 * @file
 */
// NOLINTBEGIN(readability-identifier-naming)
using TensorIndex = cudf::size_type;
using TensorSize  = std::size_t;
using RankType    = int;

using ShapeType = std::vector<TensorIndex>;
using RangeType = std::pair<TensorIndex, TensorIndex>;
using TensorMap = std::map<std::string, TensorObject>;

template <typename T = void>
using Task = mrc::coroutines::Task<T>;
// NOLINTEND(readability-identifier-naming)

/** @} */  // end of group
}  // namespace morpheus
