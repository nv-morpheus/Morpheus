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

#include "morpheus/stages/add_scores.hpp"

#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex, TensorObject

#include <glog/logging.h>
#include <mrc/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/object.hpp>          // for Object

#include <cstddef>  // for size_t
#include <exception>
#include <map>
#include <memory>
#include <ostream>      // for logging
#include <type_traits>  // for declval
#include <utility>      // for move
#include <vector>
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ AddScoresStage **************************** //
AddScoresStage::AddScoresStage(std::size_t num_class_labels,
                               std::map<std::size_t, std::string> idx2label,
                               std::string tensor_name) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_num_class_labels(num_class_labels),
  m_idx2label(std::move(idx2label)),
  m_tensor_name(std::move(tensor_name))
{
    CHECK(m_idx2label.size() <= m_num_class_labels) << "idx2label should represent a subset of the class_labels";
}

AddScoresStage::subscribe_fn_t AddScoresStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                const auto& probs = x->get_output(m_tensor_name);
                const auto& shape = probs.get_shape();

                CHECK(shape.size() == 2 && shape[1] == m_num_class_labels)
                    << "Label count does not match output of model. Label count: " << m_num_class_labels
                    << ", Model output: " << shape[1];

                const std::size_t num_rows    = shape[0];
                const std::size_t num_columns = shape[1];

                std::vector<std::string> columns(m_idx2label.size());
                std::vector<TensorObject> tensors(m_idx2label.size());

                std::size_t i = 0;
                for (const auto& [column_num, column_name] : m_idx2label)
                {
                    columns[i] = column_name;
                    tensors[i] = probs.slice(std::vector<TensorIndex>{0, static_cast<TensorIndex>(column_num)},
                                             std::vector<TensorIndex>{static_cast<TensorIndex>(num_rows),
                                                                      static_cast<TensorIndex>(column_num + 1)});

                    ++i;
                }

                x->set_meta(columns, tensors);

                output.on_next(x);
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ AddScoresStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<AddScoresStage>> AddScoresStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::size_t num_class_labels,
    std::map<std::size_t, std::string> idx2label,
    std::string tensor_name)
{
    return builder.construct_object<AddScoresStage>(
        name, num_class_labels, std::move(idx2label), std::move(tensor_name));
}
}  // namespace morpheus
