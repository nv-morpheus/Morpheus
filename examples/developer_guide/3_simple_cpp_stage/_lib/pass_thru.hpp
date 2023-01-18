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

#include <morpheus/messages/multi.hpp>  // for MultiMessage
#include <pymrc/node.hpp>               // for PythonNode
#include <mrc/segment/builder.hpp>      // for Segment Builder
#include <mrc/segment/object.hpp>       // for Segment Object

#include <memory>
#include <string>

namespace morpheus_example {

// pybind11 sets visibility to hidden by default; we want to export our symbols
#pragma GCC visibility push(default)

using namespace morpheus;

class PassThruStage : public mrc::pymrc::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};

struct PassThruStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<PassThruStage>> init(mrc::segment::Builder &builder,
                                                                     const std::string &name);
};

#pragma GCC visibility pop
}  // namespace morpheus_example
