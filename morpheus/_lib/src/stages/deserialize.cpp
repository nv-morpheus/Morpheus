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

#include "morpheus/stages/deserialize.hpp"

#include <mrc/segment/builder.hpp>
#include <pybind11/gil.h>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <algorithm>  // for min
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>  // for declval
#include <utility>

namespace {
namespace py = pybind11;
using namespace py::literals;
using namespace morpheus;

void reset_index(DeserializeStage::sink_type_t& msg)
{
    // since we are both modifying the index, and preserving the existing one as a new column, both things tracked by
    // table info we will instead copy the python object and teturn a new meta. Since this is a work-around for an issue
    // we are warning the user about correctness and safety is more important than performance.
    auto py_df = msg->get_info().copy_to_py_object();

    {
        py::gil_scoped_acquire gil;
        auto df_index   = py_df.attr("index");
        auto index_name = df_index.attr("name");

        py::str old_index_col_name{"_index_"};
        if (!index_name.is_none())
        {
            old_index_col_name += index_name;
        }

        df_index.attr("name") = old_index_col_name;

        py_df.attr("reset_index")("inplace"_a = true);
    }

    auto new_meta = MessageMeta::create_from_python(std::move(py_df));
    msg.swap(new_meta);
}

}  // namespace

namespace morpheus {
// Component public implementations
// ************ DeserializationStage **************************** //
DeserializeStage::DeserializeStage(size_t batch_size) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_batch_size(batch_size)
{}

DeserializeStage::subscribe_fn_t DeserializeStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                if (!x->has_unique_index())
                {
                    LOG(WARNING) << "Non unique index found in dataframe, generating new index.";
                    reset_index(x);
                }

                // Make one large MultiMessage
                auto full_message = std::make_shared<MultiMessage>(x, 0, x->count());

                // Loop over the MessageMeta and create sub-batches
                for (size_t i = 0; i < x->count(); i += this->m_batch_size)
                {
                    auto next = full_message->get_slice(i, std::min(i + this->m_batch_size, x->count()));

                    output.on_next(std::move(next));
                }
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ DeserializationStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<DeserializeStage>> DeserializeStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, size_t batch_size)
{
    auto stage = builder.construct_object<DeserializeStage>(name, batch_size);

    return stage;
}
}  // namespace morpheus
