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

#include "morpheus/controllers/monitor_controller.hpp"

#include "indicators/setting.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <optional>
#include <type_traits>
#include <vector>

namespace morpheus {

// Component public implementations
// ****************** MonitorController ************************ //

template <typename MessageT>
MonitorController<MessageT>::MonitorController(const std::string& description,
                                               std::optional<std::function<int(MessageT)>> determin_count_fn) :
  m_description(description),
  m_determine_count_fn(determin_count_fn),
  m_count(0)
{
    m_progress_bar.set_option(indicators::option::BarWidth{50});
    m_progress_bar.set_option(indicators::option::Start{"["});
    m_progress_bar.set_option(indicators::option::Fill("■"));
    m_progress_bar.set_option(indicators::option::Lead(">"));
    m_progress_bar.set_option(indicators::option::Remainder(" "));
    m_progress_bar.set_option(indicators::option::End("]"));
    m_progress_bar.set_option(indicators::option::PostfixText{m_description});
    m_progress_bar.set_option(indicators::option::ForegroundColor{indicators::Color::yellow});
    m_progress_bar.set_option(indicators::option::ShowElapsedTime{true});

    MonitorController::m_progress_bars.push_back(m_progress_bar);
}

template <typename MessageT>
MessageT MonitorController<MessageT>::progress_sink(MessageT msg)
{
    if (m_determine_count_fn == std::nullopt)
    {
        m_determine_count_fn = auto_count_fn(msg);
    }

    m_count += (*m_determine_count_fn)(msg);
    m_progress_bar.set_progress(m_count);

    return msg;
}

template <typename T>
struct is_vector : std::false_type
{};

template <typename T, typename U>
struct is_vector<std::vector<T, U>> : std::true_type
{};

template <typename MessageT>
auto MonitorController<MessageT>::auto_count_fn(MessageT msg) -> std::optional<std::function<int(MessageT)>>
{
    if constexpr (std::is_same_v<MessageT, cudf::table>)
    {
        return [](MessageT msg) {
            return msg.num_rows();
        };
    }

    if constexpr (std::is_same_v<MessageT, MessageMeta>)
    {
        return [](MessageT msg) {
            return msg.count();
        };
    }

    if constexpr (std::is_same_v<MessageT, ControlMessage>)
    {
        return [](MessageT msg) {
            if (!msg.payload())
            {
                return 0;
            }
            return msg.payload()->count();
        };
    }

    if constexpr (is_vector<MessageT>::value)
    {
        if (msg.empty())
        {
            return std::nullopt;
        }

        return [this](MessageT msg) {
            auto item_count_fn = auto_count_fn<MessageT>(msg[0]);
            if (item_count_fn == std::nullopt)
            {
                return 0;
            }
            return std::accumulate(msg.begin(), msg.end(), 0, [item_count_fn](int sum, const auto& item) {
                return sum + (*item_count_fn)(item);
            });
        };
    }

    throw std::runtime_error("Unsupported message type received for MonitorController");
}

template <typename MessageT>
void MonitorController<MessageT>::sink_on_completed()
{
    m_progress_bar.mark_as_completed();
}

}  // namespace morpheus
