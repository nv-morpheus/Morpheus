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

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/inference_memory_fil.hpp"
#include "morpheus/messages/memory/inference_memory_nlp.hpp"
#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_inference_fil.hpp"
#include "morpheus/messages/multi_inference_nlp.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/objects/data_table.hpp"
#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/version.hpp"

#include <boost/fiber/future/future.hpp>
#include <mrc/channel/status.hpp>  // for Status
#include <mrc/edge/edge_connector.hpp>
#include <mrc/node/port_registry.hpp>
#include <mrc/node/rx_sink_base.hpp>
#include <mrc/node/rx_source_base.hpp>
#include <mrc/types.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <pymrc/edge_adapter.hpp>
#include <pymrc/node.hpp>
#include <pymrc/port_builders.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {

namespace fs = std::filesystem;
namespace py = pybind11;

PYBIND11_MODULE(messages, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    mrc::pymrc::import(_module, "cupy");
    mrc::pymrc::import(_module, "morpheus._lib.common");

    // Required for SegmentObject
    mrc::pymrc::import(_module, "mrc.core.node");

    // Allows python objects to keep DataTable objects alive
    py::class_<IDataTable, std::shared_ptr<IDataTable>>(_module, "DataTable");

    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MessageControl>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MessageMeta>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceFILMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiInferenceNLPMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiResponseMessage>>();
    mrc::pymrc::PortBuilderUtil::register_port_util<std::shared_ptr<MultiResponseProbsMessage>>();

    // EdgeConnectors for converting between PyObjectHolders and various Message types
    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MessageControl>,
                             mrc::pymrc::PyObjectHolder>::register_converter();
    mrc::edge::EdgeConnector<mrc::pymrc::PyObjectHolder,
                             std::shared_ptr<morpheus::MessageControl>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MessageMeta>, mrc::pymrc::PyObjectHolder>::register_converter();
    mrc::edge::EdgeConnector<mrc::pymrc::PyObjectHolder, std::shared_ptr<morpheus::MessageMeta>>::register_converter();

    // EdgeConnectors for derived classes of MultiMessage to MultiMessage
    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceFILMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceNLPMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiResponseMessage>>::register_converter();

    mrc::edge::EdgeConnector<std::shared_ptr<morpheus::MultiResponseProbsMessage>,
                             std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    // TODO(Devin): Circle back on return value policy choices
    py::class_<MessageControl, std::shared_ptr<MessageControl>>(_module, "MessageControl")
        .def(py::init<>(), py::return_value_policy::reference_internal)
        .def(py::init(py::overload_cast<py::dict&>(&ControlMessageProxy::create)),
             py::return_value_policy::reference_internal)
        .def(py::init(py::overload_cast<std::shared_ptr<MessageControl>>(&ControlMessageProxy::create)),
             py::return_value_policy::reference_internal)
        .def("config",
             pybind11::overload_cast<MessageControl&>(&ControlMessageProxy::config),
             py::return_value_policy::reference_internal)
        .def("config",
             pybind11::overload_cast<MessageControl&, py::dict&>(&ControlMessageProxy::config),
             py::arg("config"))
        .def("copy", &ControlMessageProxy::copy, py::return_value_policy::reference_internal)
        .def("add_task", &ControlMessageProxy::add_task, py::arg("task_type"), py::arg("task"))
        .def("has_task", &MessageControl::has_task, py::arg("task_type"))
        .def("pop_task", &ControlMessageProxy::pop_task, py::arg("task_type"))
        .def("set_metadata", &ControlMessageProxy::set_metadata, py::arg("key"), py::arg("value"))
        .def("has_metadata", &MessageControl::has_metadata, py::arg("key"))
        .def("get_metadata", &ControlMessageProxy::get_metadata, py::arg("key"))
        .def("payload", pybind11::overload_cast<>(&MessageControl::payload), py::return_value_policy::move)
        .def("payload", pybind11::overload_cast<const std::shared_ptr<MessageMeta>&>(&MessageControl::payload));

    // Context manager for Mutable Dataframes. Attempting to use it outside a with block will raise an exception
    py::class_<MutableTableCtxMgr, std::shared_ptr<MutableTableCtxMgr>>(_module, "MutableTableCtxMgr")
        .def("__enter__", &MutableTableCtxMgr::enter, py::return_value_policy::reference)
        .def("__exit__", &MutableTableCtxMgr::exit)
        .def("__getattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__getitem__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setattr__", &MutableTableCtxMgr::throw_usage_error)
        .def("__setitem__", &MutableTableCtxMgr::throw_usage_error);

    py::class_<MessageMeta, std::shared_ptr<MessageMeta>>(_module, "MessageMeta")
        .def(py::init<>(&MessageMetaInterfaceProxy::init_python), py::arg("df"))
        .def_property_readonly("count", &MessageMetaInterfaceProxy::count)
        .def_property_readonly("df", &MessageMetaInterfaceProxy::df_property, py::return_value_policy::move)
        .def("copy_dataframe", &MessageMetaInterfaceProxy::get_data_frame, py::return_value_policy::move)
        .def("mutable_dataframe", &MessageMetaInterfaceProxy::mutable_dataframe, py::return_value_policy::move)
        .def_static("make_from_file", &MessageMetaInterfaceProxy::init_cpp);

    py::class_<MultiMessage, std::shared_ptr<MultiMessage>>(_module, "MultiMessage")
        .def(py::init<>(&MultiMessageInterfaceProxy::init),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"))
        .def_property_readonly("meta", &MultiMessageInterfaceProxy::meta)
        .def_property_readonly("mess_offset", &MultiMessageInterfaceProxy::mess_offset)
        .def_property_readonly("mess_count", &MultiMessageInterfaceProxy::mess_count)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&, std::string)>(&MultiMessageInterfaceProxy::get_meta),
             py::return_value_policy::move)
        .def("get_meta",
             static_cast<pybind11::object (*)(MultiMessage&, std::vector<std::string>)>(
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

    py::class_<InferenceMemory, std::shared_ptr<InferenceMemory>>(_module, "InferenceMemory")
        .def_property_readonly("count", &InferenceMemoryInterfaceProxy::get_count);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(_module, "InferenceMemoryNLP")
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

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(_module, "InferenceMemoryFIL")
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

    py::class_<MultiInferenceMessage, MultiMessage, std::shared_ptr<MultiInferenceMessage>>(_module,
                                                                                            "MultiInferenceMessage")
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
        _module, "MultiInferenceNLPMessage")
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
        _module, "MultiInferenceFILMessage")
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

    py::class_<TensorMemory, std::shared_ptr<TensorMemory>>(_module, "TensorMemory")
        .def_readonly("count", &TensorMemory::count);

    py::class_<ResponseMemory, std::shared_ptr<ResponseMemory>>(_module, "ResponseMemory")
        .def_readonly("count", &ResponseMemory::count)
        .def("get_output", &ResponseMemoryInterfaceProxy::get_output, py::return_value_policy::reference_internal)
        .def("get_output_tensor",
             &ResponseMemoryInterfaceProxy::get_output_tensor,
             py::return_value_policy::reference_internal);

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(_module,
                                                                                          "ResponseMemoryProbs")
        .def(py::init<>(&ResponseMemoryProbsInterfaceProxy::init), py::arg("count"), py::arg("probs"))
        .def_property_readonly("count", &ResponseMemoryProbsInterfaceProxy::count)
        .def_property(
            "probs", &ResponseMemoryProbsInterfaceProxy::get_probs, &ResponseMemoryProbsInterfaceProxy::set_probs);

    py::class_<MultiResponseMessage, MultiMessage, std::shared_ptr<MultiResponseMessage>>(_module,
                                                                                          "MultiResponseMessage")
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
        _module, "MultiResponseProbsMessage")
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

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
