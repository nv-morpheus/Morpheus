/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/objects/fiber_queue.hpp>
#include <morpheus/objects/neo_wrapped_tensor.hpp>
#include <morpheus/utilities/cudf_util.hpp>

#include <pybind11/pybind11.h>

#include <memory>

namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(common, m)
{
    google::InitGoogleLogging("morpheus");

    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.common
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    // TODO(Devin) -- This should not be defined in morpheus -- should be imported from pyneo -- wrapping for now.
    py::class_<TensorObject>(m, "Tensor")
        .def_property_readonly("__cuda_array_interface__", &NeoTensorObjectInterfaceProxy::cuda_array_interface);

    py::class_<FiberQueue, std::shared_ptr<FiberQueue>>(m, "FiberQueue")
        .def(py::init<>(&FiberQueueInterfaceProxy::init), py::arg("max_size"))
        .def("get", &FiberQueueInterfaceProxy::get, py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("put", &FiberQueueInterfaceProxy::put, py::arg("item"), py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("close", &FiberQueueInterfaceProxy::close);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
