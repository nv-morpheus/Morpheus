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

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
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

#include <pybind11/attr.h>      // for multiple_inheritance
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pybind11/stl.h>       // for dict->map conversions
#include <pysrf/utils.hpp>      // for pysrf::import
#include <srf/segment/object.hpp>

#include <memory>

namespace morpheus {
namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(stages, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.stages
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    srf::pysrf::import(m, "cupy");
    srf::pysrf::import(m, "morpheus._lib.messages");
    srf::pysrf::import(m, "morpheus._lib.file_types");

    py::class_<srf::segment::Object<AddClassificationsStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<AddClassificationsStage>>>(
        m, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<srf::segment::Object<AddScoresStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<AddScoresStage>>>(m, "AddScoresStage", py::multiple_inheritance())
        .def(py::init<>(&AddScoresStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<srf::segment::Object<DeserializeStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<DeserializeStage>>>(
        m, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("batch_size"));

    py::class_<srf::segment::Object<FileSourceStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<FileSourceStage>>>(m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>(&FileSourceStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"));

    py::class_<srf::segment::Object<FilterDetectionsStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<FilterDetectionsStage>>>(
        m, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("copy") = true);

    py::class_<srf::segment::Object<InferenceClientStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<InferenceClientStage>>>(
        m, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<srf::segment::Object<KafkaSourceStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<KafkaSourceStage>>>(
        m, "KafkaSourceStage", py::multiple_inheritance())
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

    py::class_<srf::segment::Object<PreallocateStage<MessageMeta>>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreallocateStage<MessageMeta>>>>(
        m, "PreallocateMessageMetaStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MessageMeta>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<srf::segment::Object<PreallocateStage<MultiMessage>>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreallocateStage<MultiMessage>>>>(
        m, "PreallocateMultiMessageStage", py::multiple_inheritance())
        .def(py::init<>(&PreallocateStageInterfaceProxy<MultiMessage>::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("needed_columns"));

    py::class_<srf::segment::Object<PreprocessFILStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreprocessFILStage>>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("features"));

    py::class_<srf::segment::Object<PreprocessNLPStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<PreprocessNLPStage>>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"));

    py::class_<srf::segment::Object<SerializeStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<SerializeStage>>>(m, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<srf::segment::Object<WriteToFileStage>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<WriteToFileStage>>>(
        m, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")              = "w",
             py::arg("file_type")         = 0,  // Setting this to FileTypes::AUTO throws a conversion error at runtime
             py::arg("include_index_col") = true);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
