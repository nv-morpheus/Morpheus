/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/add_classification.hpp"

#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include "morpheus/messages/control.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>  // for move
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {

// Component public implementations
// ************ AddClassificationStage **************************** //

// ************ AddClassificationStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<AddClassificationsStage<MultiResponseMessage, MultiResponseMessage>>>
AddClassificationStageInterfaceProxy::init_multi(mrc::segment::Builder& builder,
                                                 const std::string& name,
                                                 std::map<std::size_t, std::string> idx2label,
                                                 float threshold)
{
    return builder.construct_object<AddClassificationsStage<MultiResponseMessage, MultiResponseMessage>>(
        name, idx2label, threshold);
}

std::shared_ptr<mrc::segment::Object<AddClassificationsStage<ControlMessage, ControlMessage>>>
AddClassificationStageInterfaceProxy::init_cm(mrc::segment::Builder& builder,
                                              const std::string& name,
                                              std::map<std::size_t, std::string> idx2label,
                                              float threshold)
{
    return builder.construct_object<AddClassificationsStage<ControlMessage, ControlMessage>>(
        name, idx2label, threshold);
}

}  // namespace morpheus
