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

#include "mrc/channel/status.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/types.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/objects/file_types.hpp"  // for FileTypes
#include "morpheus/stages/add_classification.hpp"
#include "morpheus/stages/add_scores.hpp"
#include "morpheus/stages/deserialize.hpp"
#include "morpheus/stages/file_source.hpp"
#include "morpheus/stages/filter_detection.hpp"
#include "morpheus/stages/kafka_source.hpp"
#include "morpheus/stages/preallocate.hpp"
#include "morpheus/stages/preprocess_fil.hpp"
#include "morpheus/stages/preprocess_nlp.hpp"
#include "morpheus/stages/serialize.hpp"
#include "morpheus/stages/triton_inference.hpp"
#include "morpheus/stages/write_to_file.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/version.hpp"

#include <boost/fiber/future/future.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/attr.h>      // for multiple_inheritance
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pymrc/utils.hpp>      // for pymrc::import
#include <rxcpp/rx.hpp>

#include <map>
#include <memory>
#include <sstream>
#include <vector>

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

    mrc::pymrc::from_import(_module, "morpheus._lib.common", "FilterSource");

    py::class_<mrc::segment::Object<AddClassificationsStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddClassificationsStage>>>(
        _module, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("idx2label"),
             py::arg("threshold"));

    py::class_<mrc::segment::Object<AddScoresStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<AddScoresStage>>>(
        _module, "AddScoresStage", py::multiple_inheritance())
        .def(
            py::init<>(&AddScoresStageInterfaceProxy::init), py::arg("builder"), py::arg("name"), py::arg("idx2label"));

    py::class_<mrc::segment::Object<DeserializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<DeserializeStage>>>(
        _module, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"),
             py::arg("ensure_sliceable_index") = true);

    py::class_<mrc::segment::Object<FileSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<FileSourceStage>>>(
        _module, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>(&FileSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"));

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

    py::class_<mrc::segment::Object<InferenceClientStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<InferenceClientStage>>>(
        _module, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<mrc::segment::Object<KafkaSourceStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<KafkaSourceStage>>>(
        _module, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false,
             py::arg("stop_after")            = 0,
             py::arg("async_commits")         = true);

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

    py::class_<mrc::segment::Object<PreprocessFILStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessFILStage>>>(
        _module, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<mrc::segment::Object<PreprocessNLPStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PreprocessNLPStage>>>(
        _module, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"),
             py::arg("column"));

    py::class_<mrc::segment::Object<SerializeStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<SerializeStage>>>(
        _module, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
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
