/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <pyerrors.h>
#include <pymrc/node.hpp>
#include <pymrc/utils.hpp>  // for cast_from_pyobject
#include <rxcpp/rx.hpp>

#include <algorithm>  // for min
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

namespace morpheus {

void make_windowed_message(std::shared_ptr<MultiMessage>& full_message,
                           TensorIndex start,
                           TensorIndex stop,
                           cm_task_t* task,
                           std::shared_ptr<MultiMessage>& windowed_message)
{
    DCHECK_EQ(task, nullptr) << "Task is not supported for MultiMessage";
    auto sliced_msg = full_message->get_slice(start, stop);
    windowed_message.swap(sliced_msg);
}

void make_windowed_message(std::shared_ptr<MultiMessage>& full_message,
                           TensorIndex start,
                           TensorIndex stop,
                           cm_task_t* task,
                           std::shared_ptr<ControlMessage>& windowed_message)
{
    auto window      = full_message->copy_ranges({{start, stop}}, stop - start);
    auto new_message = std::make_shared<ControlMessage>();
    new_message->payload(window->meta);
    if (task)
    {
        new_message->add_task(task->first, task->second);
    }

    windowed_message.swap(new_message);
}

}  // namespace morpheus
