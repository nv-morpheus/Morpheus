/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"              // for MORPHEUS_EXPORT
#include "morpheus/messages/control.hpp"  // for ControlMessage

#include <mrc/node/operators/router.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <mrc/segment/object.hpp>   // for Object
#include <pymrc/edge_adapter.hpp>   // for AutoRegSinkAdapter, AutoRegSourceAdapter
#include <pymrc/node.hpp>           // IWYU pragma: keep
#include <pymrc/port_builders.hpp>  // for AutoRegEgressPort, AutoRegIngressPort
#include <pymrc/utilities/function_wrappers.hpp>
#include <rxcpp/rx.hpp>  // for decay_t, trace_activity, from, observable_member

#include <memory>  // for shared_ptr, unique_ptr
#include <string>  // for string
#include <vector>  // for vector

namespace morpheus {
/****** Component public implementations *******************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/****** RouterStage********************************/
class MORPHEUS_EXPORT RouterControlMessageComponentStage
  : public mrc::node::LambdaStaticRouterComponent<std::string, std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegSourceAdapter<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegSinkAdapter<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegIngressPort<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegEgressPort<std::shared_ptr<ControlMessage>>
{
  public:
    using mrc::node::LambdaStaticRouterComponent<std::string,
                                                 std::shared_ptr<ControlMessage>>::LambdaStaticRouterComponent;
};

class MORPHEUS_EXPORT RouterControlMessageRunnableStage
  : public mrc::node::LambdaStaticRouterRunnable<std::string, std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegSourceAdapter<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegSinkAdapter<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegIngressPort<std::shared_ptr<ControlMessage>>,
    public mrc::pymrc::AutoRegEgressPort<std::shared_ptr<ControlMessage>>
{
  public:
    using mrc::node::LambdaStaticRouterRunnable<std::string,
                                                std::shared_ptr<ControlMessage>>::LambdaStaticRouterRunnable;
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT RouterStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage that emits
     * ControlMessage's, and return the result. If `task_type` is not None,
     * `task_payload` must also be not None, and vice versa.
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage>>
     */
    static std::shared_ptr<mrc::segment::Object<RouterControlMessageComponentStage>> init_cm_component(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::vector<std::string> keys,
        mrc::pymrc::PyFuncHolder<std::string(std::shared_ptr<ControlMessage>)> key_fn);

    /**
     * @brief Create and initialize a DeserializationStage that emits
     * ControlMessage's, and return the result. If `task_type` is not None,
     * `task_payload` must also be not None, and vice versa.
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage>>
     */
    static std::shared_ptr<mrc::segment::Object<RouterControlMessageRunnableStage>> init_cm_runnable(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::vector<std::string> keys,
        mrc::pymrc::PyFuncHolder<std::string(std::shared_ptr<ControlMessage>)> key_fn);
};

/** @} */  // end of group
}  // namespace morpheus
