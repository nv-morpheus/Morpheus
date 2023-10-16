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

#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <coroutine>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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

    py::class_<BoostFibersMainPyAwaitable, CppToPyAwaitable, std::shared_ptr<BoostFibersMainPyAwaitable>>(
        _module, "BoostFibersMainPyAwaitable")
        .def(py::init<>());

    _module.def("wrap_coroutine", [](coroutines::Task<std::vector<std::string>> fn) -> coroutines::Task<std::string> {
        DCHECK_EQ(PyGILState_Check(), 0) << "Should not have the GIL when resuming a C++ coroutine";

        auto strings = co_await fn;

        co_return strings[0];
    });

    // _module.attr("__version__") =
    //     MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace mrc::pycoro
