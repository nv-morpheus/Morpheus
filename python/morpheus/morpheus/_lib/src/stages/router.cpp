/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "morpheus/stages/router.hpp"

#include "pymrc/utilities/function_wrappers.hpp"

#include <pybind11/pybind11.h>  // for cast

namespace morpheus {

namespace py = pybind11;

std::shared_ptr<mrc::segment::Object<RouterControlMessageComponentStage>> RouterStageInterfaceProxy::init_cm_component(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::vector<std::string> keys,
    mrc::pymrc::PyFuncHolder<std::string(std::shared_ptr<ControlMessage>)> key_fn)
{
    auto stage = builder.construct_object<RouterControlMessageComponentStage>(
        name, keys, [key_fn = std::move(key_fn)](const std::shared_ptr<ControlMessage>& data) {
            py::gil_scoped_acquire gil;

            auto ret_key     = key_fn(data);
            auto ret_key_str = py::str(ret_key);

            return std::string(ret_key_str);
        });

    return stage;
}

std::shared_ptr<mrc::segment::Object<RouterControlMessageRunnableStage>> RouterStageInterfaceProxy::init_cm_runnable(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::vector<std::string> keys,
    mrc::pymrc::PyFuncHolder<std::string(std::shared_ptr<ControlMessage>)> key_fn)
{
    auto stage = builder.construct_object<RouterControlMessageRunnableStage>(
        name, keys, [key_fn = std::move(key_fn)](const std::shared_ptr<ControlMessage>& data) {
            py::gil_scoped_acquire gil;

            auto ret_key     = key_fn(data);
            auto ret_key_str = py::str(ret_key);

            return std::string(ret_key_str);
        });

    return stage;
}

}  // namespace morpheus
