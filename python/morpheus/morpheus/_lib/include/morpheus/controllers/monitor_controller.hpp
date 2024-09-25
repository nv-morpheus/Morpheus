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

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/******************* MonitorController**********************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief
 */
template <typename MessageT>
class MonitorController
{
  public:
    MonitorController(const std::string& description,
                      std::optional<std::function<int(MessageT)>> determin_count_fn = std::nullopt);

  private:
    MessageT progress_sink(MessageT msg);
    auto auto_count_fn(MessageT msg) -> std::optional<std::function<int(MessageT)>>;
    void sink_on_completed();

    const std::string& m_description;
    size_t m_count;

    indicators::ProgressBar m_progress_bar;

    std::optional<std::function<int(MessageT)>> m_determine_count_fn;

    static indicators::DynamicProgress<indicators::ProgressBar> m_progress_bars;
};

template <typename InputT>
indicators::DynamicProgress<indicators::ProgressBar> MonitorController<InputT>::m_progress_bars;

/** @} */  // end of group
}  // namespace morpheus
