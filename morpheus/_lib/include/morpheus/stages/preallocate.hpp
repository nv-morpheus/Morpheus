/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/objects/dtype.hpp"  // for DType, TypeId

#include <mrc/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations *******************/
/****** PreallocateStage ********************************/
/* Preallocates new columns into the underlying dataframe. This stage supports both MessageMeta & subclasses of
 * MultiMessage. In the Python bindings the stage is bound as `PreallocateMessageMetaStage` and
 * `PreallocateMultiMessageStage`
 */
template <typename MessageT>
class PreallocateStage : public mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    PreallocateStage(const std::vector<std::tuple<std::string, TypeId>>& needed_columns);

  private:
    subscribe_fn_t build_operator();

    std::vector<std::tuple<std::string, DType>> m_needed_columns;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
template <typename MessageT>
struct PreallocateStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<PreallocateStage<MessageT>>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::vector<std::tuple<std::string, TypeId>> needed_columns);
};

// Explicit instantiations
template class PreallocateStage<MessageMeta>;
template class PreallocateStage<MultiMessage>;

template struct PreallocateStageInterfaceProxy<MessageMeta>;
template struct PreallocateStageInterfaceProxy<MultiMessage>;

#pragma GCC visibility pop
}  // namespace morpheus
