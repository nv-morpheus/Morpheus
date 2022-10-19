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

#include <morpheus/doca/doca_context.hpp>
#include <morpheus/messages/meta.hpp>

#include <cudf/io/types.hpp>  // for table_with_metadata
#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <memory>
#include <string>
#include <vector>  // for vector

namespace morpheus {
/****** Component public implementations *******************/
/****** DocaSourceStage*************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class DocaSourceStage : public srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    DocaSourceStage();

  private:
    subscriber_fn_t build();

    std::shared_ptr<morpheus::doca::doca_context> _context;
};

/****** DocaSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DocaSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DocaSourceStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<DocaSourceStage>> init(srf::segment::Builder &builder,
                                                                       const std::string &name);
};
#pragma GCC visibility pop
}  // namespace morpheus
