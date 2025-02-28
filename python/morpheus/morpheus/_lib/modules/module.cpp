/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>

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

    // Get the MRC version that we are registering these modules for. Ideally, this would be able to get it directly
    // from <mrc/version.hpp> but that file isnt exported
    std::vector<unsigned int> mrc_version;
    mrc_version.reserve(3);

    auto re  = pybind11::module_::import("re");
    auto mrc = pybind11::module_::import("mrc");
    // re.compile(r"^(\d+)\.(\d+)\.(\d+)").match(mrc.__version__).groups()
    for (const auto& x :
         re.attr("compile")(R"(^(\d+)\.(\d+)\.(\d+))").attr("match")(mrc.attr("__version__")).attr("groups")())
    {
        // int(x)
        auto i = py::int_(py::reinterpret_borrow<py::object>(x));
        mrc_version.push_back(i.cast<unsigned int>());
    }

    mrc::modules::ModelRegistryUtil::create_registered_module<DataLoaderModule>("DataLoader", "morpheus", mrc_version);

    _module.attr("__version__") =
        MORPHEUS_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
