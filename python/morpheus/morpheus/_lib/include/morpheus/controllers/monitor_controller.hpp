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
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/******************* MonitorController**********************/

/**
 * @addtogroup controllers
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
                      std::optional<std::function<size_t(MessageT)>> determine_count_fn = std::nullopt);

    auto auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>;

    MessageT progress_sink(MessageT msg);
    void sink_on_completed();

  private:
    std::unique_ptr<indicators::ProgressBar> initialize_progress_bar();
    std::unique_ptr<indicators::ProgressBar> m_progress_bar;
    const std::string& m_description;
    size_t m_count;
    std::optional<std::function<int(MessageT)>> m_determine_count_fn;
    int m_bar_id;

    static indicators::DynamicProgress<indicators::ProgressBar> s_progress_bars;
};

template <typename InputT>
indicators::DynamicProgress<indicators::ProgressBar> MonitorController<InputT>::s_progress_bars;

template <typename MessageT>
MonitorController<MessageT>::MonitorController(const std::string& description,
                                               std::optional<std::function<size_t(MessageT)>> determine_count_fn) :
  m_description(description),
  m_determine_count_fn(determine_count_fn),
  m_count(0)
{
    if (!m_determine_count_fn)
    {
        m_determine_count_fn = auto_count_fn();
        if (!m_determine_count_fn)
        {
            throw std::runtime_error("No count function provided and no default count function available");
        }
    }
    m_progress_bar = initialize_progress_bar();
    m_bar_id       = MonitorController::s_progress_bars.push_back(*m_progress_bar);
}

template <typename MessageT>
std::unique_ptr<indicators::ProgressBar> MonitorController<MessageT>::initialize_progress_bar()
{
    auto progress_bar = std::make_unique<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill("■"),
        indicators::option::Lead(">"),
        indicators::option::Remainder(" "),
        indicators::option::End("]"),
        indicators::option::PostfixText{m_description},
        indicators::option::ForegroundColor{indicators::Color::yellow},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::ShowElapsedTime{true});
    return std::move(progress_bar);
}

template <typename MessageT>
MessageT MonitorController<MessageT>::progress_sink(MessageT msg)
{
    m_count += (*m_determine_count_fn)(msg);
    MonitorController::s_progress_bars[m_bar_id].set_progress(m_count);

    return msg;
}

template <typename MessageT>
auto MonitorController<MessageT>::auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>
{
    if constexpr (std::is_same_v<MessageT, std::shared_ptr<MessageMeta>>)
    {
        return [](std::shared_ptr<MessageMeta> msg) {
            return msg->count();
        };
    }

    if constexpr (std::is_same_v<MessageT, std::vector<std::shared_ptr<MessageMeta>>>)
    {
        return [](std::vector<std::shared_ptr<MessageMeta>> msg) {
            auto item_count_fn = [](std::shared_ptr<MessageMeta> msg) {
                return msg->count();
            };
            return std::accumulate(msg.begin(), msg.end(), 0, [&](int sum, const auto& item) {
                return sum + (*item_count_fn)(item);
            });
        };
    }

    if constexpr (std::is_same_v<MessageT, std::shared_ptr<ControlMessage>>)
    {
        return [](std::shared_ptr<ControlMessage> msg) {
            if (!msg->payload())
            {
                return 0;
            }
            return msg->payload()->count();
        };
    }

    if constexpr (std::is_same_v<MessageT, std::vector<std::shared_ptr<ControlMessage>>>)
    {
        return [](std::vector<std::shared_ptr<ControlMessage>> msg) {
            auto item_count_fn = [](std::shared_ptr<ControlMessage> msg) {
                if (!msg->payload())
                {
                    return 0;
                }
                return msg->payload()->count();
            };
            return std::accumulate(msg.begin(), msg.end(), 0, [&](int sum, const auto& item) {
                return sum + (*item_count_fn)(item);
            });
        };
    }

    return std::nullopt;
}

template <typename MessageT>
void MonitorController<MessageT>::sink_on_completed()
{
    s_progress_bars[m_bar_id].mark_as_completed();
}

/** @} */  // end of group
}  // namespace morpheus
