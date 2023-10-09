/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "pycoro/pycoro.hpp"

#include "morpheus/version.hpp"

#include <pybind11/pybind11.h>

namespace mrc::pycoro {

namespace py = pybind11;

PYBIND11_MODULE(pycoro, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    py::class_<CppToPyAwaitable, std::shared_ptr<CppToPyAwaitable>>(_module, "CppToPyAwaitable")
        .def(py::init<>())
        .def("__iter__", &CppToPyAwaitable::iter)
        .def("__await__", &CppToPyAwaitable::await)
        .def("__next__", &CppToPyAwaitable::next);

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace mrc::pycoro
