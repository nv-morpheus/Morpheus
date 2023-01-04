/**
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

#include "morpheus/objects/filter_source.hpp"

#include <pybind11/pybind11.h>

namespace morpheus {

namespace py = pybind11;

PYBIND11_MODULE(filter_source, m)
{
    py::enum_<FilterSource>(
        m, "FilterSource", "Enum to indicate which source the FilterDetectionsStage should operate on.")
        .value("AUTO", FilterSource::AUTO)
        .value("TENSOR", FilterSource::TENSOR)
        .value("DATAFRAME", FilterSource::DATAFRAME);

}  // module
}  // namespace morpheus
