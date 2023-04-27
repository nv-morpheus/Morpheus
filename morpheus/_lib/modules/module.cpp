/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/modules/data_loader_module.hpp"
#include "morpheus/utilities/string_util.hpp"
#include "morpheus/version.hpp"

#include <mrc/modules/module_registry_util.hpp>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11

#include <sstream>
#include <vector>

namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(modules, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.modules
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    const std::vector<unsigned int> MorpheusModuleVersion{
        morpheus_VERSION_MAJOR, morpheus_VERSION_MINOR, morpheus_VERSION_PATCH};

    mrc::modules::ModelRegistryUtil::create_registered_module<DataLoaderModule>(
        "DataLoader", "morpheus", MorpheusModuleVersion);

    _module.attr("__version__") =
        MORPHEUS_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
