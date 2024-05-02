/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/file_types.hpp"
#include "morpheus/stages/add_classification.hpp"
#include "morpheus/stages/add_scores.hpp"
#include "morpheus/stages/deserialize.hpp"
#include "morpheus/stages/file_source.hpp"
#include "morpheus/stages/filter_detection.hpp"
#include "morpheus/stages/http_server_source_stage.hpp"
#include "morpheus/stages/inference_client_stage.hpp"
#include "morpheus/stages/kafka_source.hpp"
#include "morpheus/stages/preallocate.hpp"
#include "morpheus/stages/preprocess_fil.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"
#include "morpheus/stages/serialize.hpp"
#include "morpheus/stages/write_to_file.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/http_server.hpp"
#include "morpheus/version.hpp"

#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/attr.h>            // for multiple_inheritance
#include <pybind11/pybind11.h>        // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>         // for dict, sequence
#include <pybind11/stl/filesystem.h>  // IWYU pragma: keep
#include <pymrc/utils.hpp>            // for pymrc::import
#include <rxcpp/rx.hpp>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

namespace morpheus {
namespace py = pybind11;

PYBIND11_MODULE(stages, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    CudfHelper::load();

    // Make sure to load mrc.core.segment to get ObjectProperties
    mrc::pymrc::import(_module, "mrc.core.segment");

    mrc::pymrc::from_import(_module, "morpheus._lib.common", "FilterSource");

    py::class_<mrc::segment::Object<AddClassificationsStageMM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddClassificationsStageMM>>>(
        _module, "AddClassificationsMultiResponseMessageStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init_multi),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"),
             py::arg("threshold"));

    py::class_<mrc::segment::Object<AddClassificationsStageCM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddClassificationsStageCM>>>(
        _module, "AddClassificationsControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"),
             py::arg("threshold"));

    py::class_<mrc::segment::Object<AddScoresStageMM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddScoresStageMM>>>(
        _module, "AddScoresMultiResponseMessageStage", py::multiple_inheritance())
        .def(py::init<>(&AddScoresStageInterfaceProxy::init_multi),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"));

    py::class_<mrc::segment::Object<AddScoresStageCM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddScoresStageCM>>>(
        _module, "AddScoresControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&AddScoresStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"));

    py::class_<mrc::segment::Object<DeserializeStage<MultiMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<DeserializeStage<MultiMessage>>>>(
        _module, "DeserializeMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init_multi),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"),
             py::arg("ensure_sliceable_index") = true);

    py::class_<mrc::segment::Object<DeserializeStage<ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<DeserializeStage<ControlMessage>>>>(
        _module, "DeserializeControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"),
             py::arg("ensure_sliceable_index") = true,
             py::arg("task_type")              = py::none(),
             py::arg("task_payload")           = py::none());

    py::class_<mrc::segment::Object<FileSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FileSourceStage>>>(
        _module, "FileSourceStage", py::multiple_inheritance())
        .def(py::init(py::overload_cast<mrc::segment::Builder&, const std::string&, std::string, int, py::dict>(
                 &FileSourceStageInterfaceProxy::init)),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"),
             py::arg("parser_kwargs"))
        .def(py::init(
                 py::overload_cast<mrc::segment::Builder&, const std::string&, std::filesystem::path, int, py::dict>(
                     &FileSourceStageInterfaceProxy::init)),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"),
             py::arg("parser_kwargs"));

    py::class_<mrc::segment::Object<FilterDetectionsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FilterDetectionsStage>>>(
        _module, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("copy"),
             py::arg("filter_source"),
             py::arg("field_name") = "probs");

    py::class_<
        mrc::segment::Object<InferenceClientStage<MultiInferenceMessage, MultiResponseMessage>>,
        mrc::segment::ObjectProperties,
        std::shared_ptr<mrc::segment::Object<InferenceClientStage<MultiInferenceMessage, MultiResponseMessage>>>>(
        _module, "InferenceClientStageMM", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init_mm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("server_url"),
             py::arg("model_name"),
             py::arg("needs_logits"),
             py::arg("force_convert_inputs"),
             py::arg("input_mapping")  = py::dict(),
             py::arg("output_mapping") = py::dict());

    py::class_<mrc::segment::Object<InferenceClientStage<ControlMessage, ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<InferenceClientStage<ControlMessage, ControlMessage>>>>(
        _module, "InferenceClientStageCM", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("server_url"),
             py::arg("model_name"),
             py::arg("needs_logits"),
             py::arg("force_convert_inputs"),
             py::arg("input_mapping")  = py::dict(),
             py::arg("output_mapping") = py::dict());

    py::class_<mrc::segment::Object<KafkaSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<KafkaSourceStage>>>(
        _module, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_single_topic),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true,
             py::arg("oauth_callback")        = py::none())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init_with_multiple_topics),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topics"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true,
             py::arg("oauth_callback")        = py::none());

    py::class_<mrc::segment::Object<PreallocateStage<ControlMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<ControlMessage>>>>(
        _module, "PreallocateControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<ControlMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreallocateStage<MessageMeta>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageMeta>>>>(
        _module, "PreallocateMessageMetaStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MessageMeta>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreallocateStage<MultiMessage>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreallocateStage<MultiMessage>>>>(
        _module, "PreallocateMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MultiMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<mrc::segment::Object<PreprocessFILStageMM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessFILStageMM>>>(
        _module, "PreprocessFILMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init_multi),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<mrc::segment::Object<PreprocessFILStageCM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessFILStageCM>>>(
        _module, "PreprocessFILControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<mrc::segment::Object<PreprocessNLPStageMM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessNLPStageMM>>>(
        _module, "PreprocessNLPMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init_multi),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"),
             py::arg("column"));

    py::class_<mrc::segment::Object<PreprocessNLPStageCM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessNLPStageCM>>>(
        _module, "PreprocessNLPControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"),
             py::arg("column"));

    py::class_<mrc::segment::Object<HttpServerSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<HttpServerSourceStage>>>(
        _module, "HttpServerSourceStage", py::multiple_inheritance())
        .def(py::init<>(&HttpServerSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("bind_address")       = "127.0.0.1",
             py::arg("port")               = 8080,
             py::arg("endpoint")           = "/message",
             py::arg("method")             = "POST",
             py::arg("accept_status")      = 201u,
             py::arg("sleep_time")         = 0.1f,
             py::arg("queue_timeout")      = 5,
             py::arg("max_queue_size")     = 1024,
             py::arg("num_server_threads") = 1,
             py::arg("max_payload_size")   = DefaultMaxPayloadSize,
             py::arg("request_timeout")    = 30,
             py::arg("lines")              = false,
             py::arg("stop_after")         = 0);

    py::class_<mrc::segment::Object<SerializeStageMM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<SerializeStageMM>>>(
        _module, "SerializeMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init_mm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<mrc::segment::Object<SerializeStageCM>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<SerializeStageCM>>>(
        _module, "SerializeControlMessageStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init_cm),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<mrc::segment::Object<WriteToFileStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<WriteToFileStage>>>(
        _module, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")              = "w",
             py::arg("file_type")         = FileTypes::Auto,
             py::arg("include_index_col") = true,
             py::arg("flush")             = false);

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus
