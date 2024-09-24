/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"                       // for MORPHEUS_EXPORT
#include "morpheus/stages/monitor_controller.hpp"  // for MonitorController

#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** MonitorStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief
 */
template <typename MessageT>
class MORPHEUS_EXPORT MonitorStage : public mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageT>, std::shared_ptr<MessageT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    MonitorStage(const std::string& description,
                 float smoothing,
                 const std::string& unit,
                 bool delayed_start,
                 std::optional<std::function<int(MessageT)>> determine_count_fn = std::nullopt);

  private:
    subscribe_fn_t build_operator();

    MonitorController<MessageT> m_monitor_controller;
};

/****** MonitorStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT MonitorStageInterfaceProxy
{
    template <typename MessageT>
    static std::shared_ptr<mrc::segment::Object<MonitorStage<MessageT>>> init(mrc::segment::Builder& builder);
};

/** @} */  // end of group
}  // namespace morpheus
