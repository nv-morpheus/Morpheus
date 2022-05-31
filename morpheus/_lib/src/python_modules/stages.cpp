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

#include <morpheus/stages/add_classification.hpp>
#include <morpheus/stages/add_scores.hpp>
#include <morpheus/stages/deserialization.hpp>
#include <morpheus/stages/file_source.hpp>
#include <morpheus/stages/filter_detection.hpp>
#include <morpheus/stages/kafka_source.hpp>
#include <morpheus/stages/preprocess_fil.hpp>
#include <morpheus/stages/preprocess_nlp.hpp>
#include <morpheus/stages/serialize.hpp>
#include <morpheus/stages/triton_inference.hpp>
#include <morpheus/stages/write_to_file.hpp>
#include <morpheus/utilities/cudf_util.hpp>

#include <neo/segment/object.hpp>

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

    neo::pyneo::import(m, "cupy");
    neo::pyneo::import(m, "morpheus._lib.messages");
    neo::pyneo::import(m, "morpheus._lib.file_types");

    py::class_<neo::segment::Object<AddClassificationsStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<AddClassificationsStage>>>(
        m, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>(&AddClassificationStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<neo::segment::Object<AddScoresStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<AddScoresStage>>>(m, "AddScoresStage", py::multiple_inheritance())
        .def(py::init<>(&AddScoresStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<neo::segment::Object<DeserializeStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<DeserializeStage>>>(
        m, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>(&DeserializeStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("batch_size"));

    py::class_<neo::segment::Object<FileSourceStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<FileSourceStage>>>(m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>(&FileSourceStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"));

    py::class_<neo::segment::Object<FilterDetectionsStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<FilterDetectionsStage>>>(
        m, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>(&FilterDetectionStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("threshold"));

    py::class_<neo::segment::Object<InferenceClientStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<InferenceClientStage>>>(
        m, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>(&InferenceClientStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<neo::segment::Object<KafkaSourceStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<KafkaSourceStage>>>(
        m, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&KafkaSourceStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits")       = false,
             py::arg("disable_pre_filtering") = false);

    py::class_<neo::segment::Object<PreprocessFILStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<PreprocessFILStage>>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessFILStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("features"));

    py::class_<neo::segment::Object<PreprocessNLPStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<PreprocessNLPStage>>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>(&PreprocessNLPStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"));

    py::class_<neo::segment::Object<SerializeStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<SerializeStage>>>(m, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>(&SerializeStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<neo::segment::Object<WriteToFileStage>,
               neo::segment::ObjectProperties,
               std::shared_ptr<neo::segment::Object<WriteToFileStage>>>(
        m, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>(&WriteToFileStageInterfaceProxy::init),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")      = "w",
             py::arg("file_type") = 0);  // Setting this to FileTypes::AUTO throws a conversion error at runtime

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
