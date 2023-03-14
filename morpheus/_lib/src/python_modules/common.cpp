/**
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

#include "morpheus/objects/dtype.hpp"  // for TypeId
#include "morpheus/objects/fiber_queue.hpp"
#include "morpheus/objects/file_types.hpp"
#include "morpheus/objects/filter_source.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorObject
#include "morpheus/objects/wrapped_tensor.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(common, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.common
        .. autosummary::
           :toctree: _generate
        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    py::class_<TensorObject>(m, "Tensor")
        .def_property_readonly("__cuda_array_interface__", &TensorObjectInterfaceProxy::cuda_array_interface);

    py::class_<FiberQueue, std::shared_ptr<FiberQueue>>(m, "FiberQueue")
        .def(py::init<>(&FiberQueueInterfaceProxy::init), py::arg("max_size"))
        .def("get", &FiberQueueInterfaceProxy::get, py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("put", &FiberQueueInterfaceProxy::put, py::arg("item"), py::arg("block") = true, py::arg("timeout") = 0.0)
        .def("close", &FiberQueueInterfaceProxy::close);

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

    m.def("tyepid_to_numpy_str", [](TypeId tid) { return DType(tid).type_str(); });

    py::enum_<FileTypes>(m,
                         "FileTypes",
                         "The type of files that the `FileSourceStage` can read and `WriteToFileStage` can write. Use "
                         "'auto' to determine from the file extension.")
        .value("Auto", FileTypes::Auto)
        .value("JSON", FileTypes::JSON)
        .value("CSV", FileTypes::CSV);

    m.def("determine_file_type", &FileTypesInterfaceProxy::determine_file_type);

    py::enum_<FilterSource>(
        m, "FilterSource", "Enum to indicate which source the FilterDetectionsStage should operate on.")
        .value("Auto", FilterSource::Auto)
        .value("TENSOR", FilterSource::TENSOR)
        .value("DATAFRAME", FilterSource::DATAFRAME);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
