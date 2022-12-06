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

#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>
#include <mrc/channel/status.hpp>          // for Status
#include <mrc/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object

#include <cstddef>  // for size_t
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** DeserializationStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief Slices incoming Dataframes into smaller `batch_size`'d chunks. This stage accepts the `MessageMeta` output
 * from `FileSourceStage`/`KafkaSourceStage` stages breaking them up into into `MultiMessage`'s. This should be one of
 * the first stages after the `Source` object.
 */
class DeserializeStage : public mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MultiMessage>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Deserialize Stage object
     *
     * @param batch_size : Number of messages to be divided into each batch
     */
    DeserializeStage(size_t batch_size);

  private:
    /**
     * TODO(Documentation)
     */
    subscribe_fn_t build_operator();

    size_t m_batch_size;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DeserializeStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param batch_size : Number of messages to be divided into each batch
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage>>
     */
    static std::shared_ptr<mrc::segment::Object<DeserializeStage>> init(mrc::segment::Builder &builder,
                                                                        const std::string &name,
                                                                        size_t batch_size);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
