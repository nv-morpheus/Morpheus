/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/memory/inference_memory_nlp.hpp"
#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_inference_fil.hpp"
#include "morpheus/messages/multi_inference_nlp.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <pybind11/cast.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <srf/node/edge_connector.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace morpheus {

namespace fs = std::filesystem;
namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(messages, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    srf::pysrf::import(m, "cupy");
    srf::pysrf::import(m, "morpheus._lib.common");

    // Required for SegmentObject
    srf::pysrf::import(m, "srf.core.node");

    // Allows python objects to keep DataTable objects alive
    py::class_<IDataTable, std::shared_ptr<IDataTable>>(m, "DataTable");

    // EdgeConnectors for derived classes of MultiMessage to MultiMessage
    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiResponseMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiResponseMessage>>::register_converter();

    srf::node::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    py::class_<MessageMeta, std::shared_ptr<MessageMeta>>(m, "MessageMeta")
        .def(py::init<>(&MessageMetaInterfaceProxy::init_python), py::arg("df"))
        .def_property_readonly("count", &MessageMetaInterfaceProxy::count)
        .def_property_readonly("df", &MessageMetaInterfaceProxy::get_data_frame, py::return_value_policy::move)
        .def_static("make_from_file", &MessageMetaInterfaceProxy::init_cpp);

    py::class_<MultiMessage, std::shared_ptr<MultiMessage>>(m, "MultiMessage")
        .def(py::init<>(&MultiMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"))
        .def_property_readonly("meta", &MultiMessageInterfaceProxy::meta)
        .def_property_readonly("mess_offset", &MultiMessageInterfaceProxy::mess_offset)
        .def_property_readonly("mess_count", &MultiMessageInterfaceProxy::mess_count)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage &)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage &, std::string)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage &, std::vector<std::string>)>(
                 &MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("set_meta", &MultiMessageInterfaceProxy::set_meta, py::return_value_policy::move)
        .def("get_slice", &MultiMessageInterfaceProxy::get_slice, py::return_value_policy::reference_internal)
        .def("copy_ranges",
             &MultiMessageInterfaceProxy::copy_ranges,
             py::arg("ranges"),
             py::arg("num_selected_rows") = py::none(),
             py::return_value_policy::move)
        .def("get_meta_list", &MultiMessageInterfaceProxy::get_meta_list, py::return_value_policy::move);

    py::class_<InferenceMemory, std::shared_ptr<InferenceMemory>>(m, "InferenceMemory")
        .def_property_readonly("count", &InferenceMemoryInterfaceProxy::get_count);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(m, "InferenceMemoryNLP")
        .def(py::init<>(&InferenceMemoryNLPInterfaceProxy::init),
             py::arg("count"),
             py::arg("input_ids"),
             py::arg("input_mask"),
             py::arg("seq_ids"))
        .def_property_readonly("count", &InferenceMemoryNLPInterfaceProxy::count)
        .def_property("input_ids",
                      &InferenceMemoryNLPInterfaceProxy::get_input_ids,
                      &InferenceMemoryNLPInterfaceProxy::set_input_ids)
        .def_property("input_mask",
                      &InferenceMemoryNLPInterfaceProxy::get_input_mask,
                      &InferenceMemoryNLPInterfaceProxy::set_input_mask)
        .def_property(
            "seq_ids", &InferenceMemoryNLPInterfaceProxy::get_seq_ids, &InferenceMemoryNLPInterfaceProxy::set_seq_ids);

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(m, "InferenceMemoryFIL")
        .def(py::init<>(&InferenceMemoryFILInterfaceProxy::init),
             py::arg("count"),
             py::arg("input__0"),
             py::arg("seq_ids"))
        .def_property_readonly("count", &InferenceMemoryFILInterfaceProxy::count)
        .def("get_tensor", &InferenceMemoryFILInterfaceProxy::get_tensor)
        .def_property("input__0",
                      &InferenceMemoryFILInterfaceProxy::get_input__0,
                      &InferenceMemoryFILInterfaceProxy::set_input__0)
        .def_property(
            "seq_ids", &InferenceMemoryFILInterfaceProxy::get_seq_ids, &InferenceMemoryFILInterfaceProxy::set_seq_ids);

    py::class_<MultiInferenceMessage, MultiMessage, std::shared_ptr<MultiInferenceMessage>>(m, "MultiInferenceMessage")
        .def(py::init<>(&MultiInferenceMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_property_readonly("memory", &MultiInferenceMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiInferenceMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiInferenceMessageInterfaceProxy::count)
        .def("get_input", &MultiInferenceMessageInterfaceProxy::get_input)
        .def("get_slice", &MultiInferenceMessageInterfaceProxy::get_slice, py::return_value_policy::reference_internal);

    py::class_<MultiInferenceNLPMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceNLPMessage>>(
        m, "MultiInferenceNLPMessage")
        .def(py::init<>(&MultiInferenceNLPMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_property_readonly("memory", &MultiInferenceNLPMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiInferenceNLPMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiInferenceNLPMessageInterfaceProxy::count)
        .def_property_readonly("input_ids", &MultiInferenceNLPMessageInterfaceProxy::input_ids)
        .def_property_readonly("input_mask", &MultiInferenceNLPMessageInterfaceProxy::input_mask)
        .def_property_readonly("seq_ids", &MultiInferenceNLPMessageInterfaceProxy::seq_ids);

    py::class_<MultiInferenceFILMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceFILMessage>>(
        m, "MultiInferenceFILMessage")
        .def(py::init<>(&MultiInferenceFILMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_property_readonly("memory", &MultiInferenceFILMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiInferenceFILMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiInferenceFILMessageInterfaceProxy::count);

    py::class_<TensorMemory, std::shared_ptr<TensorMemory>>(m, "TensorMemory")
        .def_readonly("count", &TensorMemory::count);

    py::class_<ResponseMemory, std::shared_ptr<ResponseMemory>>(m, "ResponseMemory")
        .def_readonly("count", &ResponseMemory::count)
        .def("get_output", &ResponseMemoryInterfaceProxy::get_output, py::return_value_policy::reference_internal)
        .def("get_output_tensor",
             &ResponseMemoryInterfaceProxy::get_output_tensor,
             py::return_value_policy::reference_internal);

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(m, "ResponseMemoryProbs")
        .def(py::init<>(&ResponseMemoryProbsInterfaceProxy::init), py::arg("count"), py::arg("probs"))
        .def_property_readonly("count", &ResponseMemoryProbsInterfaceProxy::count)
        .def_property(
            "probs", &ResponseMemoryProbsInterfaceProxy::get_probs, &ResponseMemoryProbsInterfaceProxy::set_probs);

    py::class_<MultiResponseMessage, MultiMessage, std::shared_ptr<MultiResponseMessage>>(m, "MultiResponseMessage")
        .def(py::init<>(&MultiResponseMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_property_readonly("memory", &MultiResponseMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiResponseMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiResponseMessageInterfaceProxy::count)
        .def("get_output", &MultiResponseMessageInterfaceProxy::get_output);

    py::class_<MultiResponseProbsMessage, MultiResponseMessage, std::shared_ptr<MultiResponseProbsMessage>>(
        m, "MultiResponseProbsMessage")
        .def(py::init<>(&MultiResponseProbsMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_property_readonly("memory", &MultiResponseProbsMessageInterfaceProxy::memory)
        .def_property_readonly("offset", &MultiResponseProbsMessageInterfaceProxy::offset)
        .def_property_readonly("count", &MultiResponseProbsMessageInterfaceProxy::count)
        .def_property_readonly("probs", &MultiResponseProbsMessageInterfaceProxy::probs);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
