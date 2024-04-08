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

#include "morpheus/stages/deserialize.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/types.hpp"

#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>  // for cast_from_pyobject
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"

namespace morpheus {

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         TensorIndex start,
                         TensorIndex stop,
                         cm_task_t* task,
                         std::shared_ptr<MultiMessage>& windowed_message)
{
    DCHECK_EQ(task, nullptr) << "Task is not supported for MultiMessage";
    auto sliced_msg = std::make_shared<MultiMessage>(incoming_message, start, stop - start);
    windowed_message.swap(sliced_msg);
}

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         TensorIndex start,
                         TensorIndex stop,
                         cm_task_t* task,
                         std::shared_ptr<ControlMessage>& windowed_message)
{
    auto slidced_meta = std::make_shared<SlicedMessageMeta>(incoming_message, start, stop);
    auto message      = std::make_shared<ControlMessage>();
    message->payload(slidced_meta);
    if (task)
    {
        message->add_task(task->first, task->second);
    }

    windowed_message.swap(message);
}

std::shared_ptr<mrc::segment::Object<DeserializeStage<MultiMessage>>> DeserializeStageInterfaceProxy::init_multi(
    mrc::segment::Builder& builder, const std::string& name, TensorIndex batch_size, bool ensure_sliceable_index)
{
    return builder.construct_object<DeserializeStage<MultiMessage>>(name, batch_size, ensure_sliceable_index, nullptr);
}

std::shared_ptr<mrc::segment::Object<DeserializeStage<ControlMessage>>> DeserializeStageInterfaceProxy::init_cm(
    mrc::segment::Builder& builder,
    const std::string& name,
    TensorIndex batch_size,
    bool ensure_sliceable_index,
    const pybind11::object& task_type,
    const pybind11::object& task_payload)
{
    std::unique_ptr<cm_task_t> task{nullptr};

    if (!task_type.is_none() && !task_payload.is_none())
    {
        task = std::make_unique<cm_task_t>(pybind11::cast<std::string>(task_type),
                                           mrc::pymrc::cast_from_pyobject(task_payload));
    }

    auto stage = builder.construct_object<DeserializeStage<ControlMessage>>(
        name, batch_size, ensure_sliceable_index, std::move(task));

    return stage;
}

}  // namespace morpheus
