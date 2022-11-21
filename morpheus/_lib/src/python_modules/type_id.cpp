/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/dtype.hpp"

#include <pybind11/pybind11.h>

namespace morpheus {

namespace py = pybind11;

PYBIND11_MODULE(type_id, m)
{
    py::enum_<TypeId>(m, "TypeId", "Supported Morpheus types")
        .value("EMPTY", TypeId::EMPTY)
        .value("INT8", TypeId::INT8)
        .value("INT16", TypeId::INT16)
        .value("INT32", TypeId::INT32)
        .value("INT64", TypeId::INT64)
        .value("UINT8", TypeId::UINT8)
        .value("UINT16", TypeId::UINT16)
        .value("UINT32", TypeId::UINT32)
        .value("UINT64", TypeId::UINT64)
        .value("FLOAT32", TypeId::FLOAT32)
        .value("FLOAT64", TypeId::FLOAT64)
        .value("BOOL8", TypeId::BOOL8)
        .value("STRING", TypeId::STRING);

}  // module
}  // namespace morpheus
