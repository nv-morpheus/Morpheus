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

#pragma once

#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <cstddef>  // for size_t
#include <map>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** PreallocateStage ********************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class PreallocateStage : public srf::pysrf::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreallocateStage(const std::map<std::string, std::string> &needed_columns);

  private:
    subscribe_fn_t build_operator();

    std::vector<std::string> m_column_names;
    std::vector<TypeId> m_column_types;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct PreallocateStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<PreallocateStage>> init(
        srf::segment::Builder &builder, const std::string &name, std::map<std::string, std::string> needed_columns);
};
#pragma GCC visibility pop
}  // namespace morpheus
