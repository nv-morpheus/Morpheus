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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/types.hpp"                  // for TensorIndex
#include "morpheus/utilities/python_util.hpp"  // for show_warning_message
#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <nlohmann/json.hpp>
#include <pybind11/pytypes.h>  // for object
#include <pyerrors.h>          // for PyExc_RuntimeWarning
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <algorithm>  // IWYU pragma: keep for std::min
#include <exception>  // for exception_ptr
#include <memory>
#include <sstream>  // IWYU pragma: keep for glog
#include <string>
#include <utility>  // for pair

namespace morpheus {
/****** Component public implementations *******************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

using cm_task_t = std::pair<std::string, nlohmann::json>;

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         TensorIndex start,
                         TensorIndex stop,
                         cm_task_t* task,
                         std::shared_ptr<MultiMessage>& windowed_message);

void make_output_message(std::shared_ptr<MessageMeta>& incoming_message,
                         TensorIndex start,
                         TensorIndex stop,
                         cm_task_t* task,
                         std::shared_ptr<ControlMessage>& windowed_message);

/****** DeserializationStage********************************/
template <typename OutputT>
class MORPHEUS_EXPORT DeserializeStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<OutputT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Deserialize Stage object
     *
     * @param batch_size Number of messages to be divided into each batch
     * @param ensure_sliceable_index Whether or not to call `ensure_sliceable_index()` on all incoming `MessageMeta`
     * @param task Optional task to be added to all outgoing `ControlMessage`s, ignored when `OutputT` is `MultiMessage`
     */
    DeserializeStage(TensorIndex batch_size,
                     bool ensure_sliceable_index     = true,
                     std::unique_ptr<cm_task_t> task = nullptr) :
      base_t(base_t::op_factory_from_sub_fn(build_operator())),
      m_batch_size(batch_size),
      m_ensure_sliceable_index(ensure_sliceable_index),
      m_task(std::move(task)){};

  private:
    subscribe_fn_t build_operator();

    TensorIndex m_batch_size;
    bool m_ensure_sliceable_index{true};
    std::unique_ptr<cm_task_t> m_task{nullptr};
};

/****** DeserializationStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT DeserializeStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DeserializationStage that emits MultiMessage's, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param batch_size : Number of messages to be divided into each batch
     * @param ensure_sliceable_index Whether or not to call `ensure_sliceable_index()` on all incoming `MessageMeta`
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage<MultiMessage>>>
     */
    static std::shared_ptr<mrc::segment::Object<DeserializeStage<MultiMessage>>> init_multi(
        mrc::segment::Builder& builder, const std::string& name, TensorIndex batch_size, bool ensure_sliceable_index);

    /**
     * @brief Create and initialize a DeserializationStage that emits ControlMessage's, and return the result.
     * If `task_type` is not None, `task_payload` must also be not None, and vice versa.
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param batch_size : Number of messages to be divided into each batch
     * @param ensure_sliceable_index Whether or not to call `ensure_sliceable_index()` on all incoming `MessageMeta`
     * @param task_type : Optional task type to be added to all outgoing messages
     * @param task_payload : Optional json object describing the task to be added to all outgoing messages
     * @return std::shared_ptr<mrc::segment::Object<DeserializeStage<ControlMessage>>>
     */
    static std::shared_ptr<mrc::segment::Object<DeserializeStage<ControlMessage>>> init_cm(
        mrc::segment::Builder& builder,
        const std::string& name,
        TensorIndex batch_size,
        bool ensure_sliceable_index,
        const pybind11::object& task_type,
        const pybind11::object& task_payload);
};

template <typename OutputT>
typename DeserializeStage<OutputT>::subscribe_fn_t DeserializeStage<OutputT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t incoming_message) {
                if (!incoming_message->has_sliceable_index())
                {
                    if (m_ensure_sliceable_index)
                    {
                        auto old_index_name = incoming_message->ensure_sliceable_index();

                        if (old_index_name.has_value())
                        {
                            // Generate a warning
                            LOG(WARNING) << MORPHEUS_CONCAT_STR(
                                "Incoming MessageMeta does not have a unique and monotonic index. Updating index "
                                "to be unique. Existing index will be retained in column '"
                                << *old_index_name << "'");
                        }
                    }
                    else
                    {
                        utilities::show_warning_message(
                            "Detected a non-sliceable index on an incoming MessageMeta. Performance when taking slices "
                            "of messages may be degraded. Consider setting `ensure_sliceable_index==True`",
                            PyExc_RuntimeWarning);
                    }
                }
                // Loop over the MessageMeta and create sub-batches
                for (TensorIndex i = 0; i < incoming_message->count(); i += this->m_batch_size)
                {
                    std::shared_ptr<OutputT> windowed_message{nullptr};
                    make_output_message(incoming_message,
                                        i,
                                        std::min(i + this->m_batch_size, incoming_message->count()),
                                        m_task.get(),
                                        windowed_message);
                    output.on_next(std::move(windowed_message));
                }
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}
/** @} */  // end of group
}  // namespace morpheus
