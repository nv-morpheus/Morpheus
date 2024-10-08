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
#include "indicators/setting.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/line.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <indicators/color.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/font_style.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace morpheus {
/******************* MonitorController**********************/

/**
 * @addtogroup controllers
 * @{
 * @file
 */

struct LineInsertingFilter : boost::iostreams::line_filter
{
    std::string do_filter(const std::string& line)
    {
        return "\n\033[A\033[1L" + line;
    }
};

// A singleton that manages the lifetime of progress bars related to any MonitorController<T> instances
class ProgressBarContextManager
{
  public:
    static ProgressBarContextManager& get_instance()
    {
        static ProgressBarContextManager instance;
        return instance;
    }
    ProgressBarContextManager(ProgressBarContextManager const&)            = delete;
    ProgressBarContextManager& operator=(ProgressBarContextManager const&) = delete;

    size_t add_progress_bar(const std::string& description,
                            indicators::Color font_color     = indicators::Color::cyan,
                            indicators::FontStyle font_style = indicators::FontStyle::bold)
    {
        m_progress_bars.push_back(std::move(initialize_progress_bar(description, font_color, font_style)));

        // DynamicProgress should take ownership over progressbars: https://github.com/p-ranav/indicators/issues/134
        // The fix to this issue is not yet released, so we need to:
        //     - Maintain the lifetime of the progress bar in m_progress_bars
        //     - Push the underlying progress bar object to the DynamicProgress container, since it accepts
        //        Indicator &bar rather than std::unique_ptr<Indicator> bar before the fix
        return m_dynamic_progress_bars.push_back(*m_progress_bars.back());
    }

    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>>& progress_bars()
    {
        return m_progress_bars;
    }

    void display()
    {
        m_std_os.iword(termcolor::_internal::colorize_index()) = 1;
        for (auto& pbar : m_progress_bars)
        {
            pbar->print_progress(true);
            m_std_os << termcolor::reset;
            m_std_os << "\n";
        }
        m_std_os << "\033[" << m_progress_bars.size() << "A";
    }

    std::ostream& std_os()
    {
        return m_std_os;
    }

  private:
    ProgressBarContextManager() : m_std_streambuf(std::cout.rdbuf()), m_std_os(m_std_streambuf)
    {
        init_log_buf();
    }
    ~ProgressBarContextManager()
    {
        std::cout.rdbuf(m_std_streambuf);
    }

    void init_log_buf()
    {
        m_log_streambuf.push(LineInsertingFilter());
        m_log_streambuf.push(*m_std_streambuf);
        std::cout.rdbuf(&m_log_streambuf);
    }

    std::unique_ptr<indicators::IndeterminateProgressBar> initialize_progress_bar(const std::string& description,
                                                                                  indicators::Color font_color,
                                                                                  indicators::FontStyle font_style)
    {
        auto progress_bar = std::make_unique<indicators::IndeterminateProgressBar>(
            indicators::option::BarWidth{10},
            indicators::option::Start{"["},
            indicators::option::Fill{" "},
            indicators::option::Lead{"#"},
            indicators::option::End("]"),
            indicators::option::PrefixText{description},
            indicators::option::ForegroundColor{font_color},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{font_style}},
            indicators::option::Stream{m_std_os});

        return std::move(progress_bar);
    }

    indicators::DynamicProgress<indicators::IndeterminateProgressBar> m_dynamic_progress_bars;
    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>> m_progress_bars;

    std::streambuf* m_std_streambuf;
    std::ostream m_std_os;
    boost::iostreams::filtering_ostreambuf m_log_streambuf;
};

/**
 * @brief A controller class that manages the display of progress bars that used by MonitorStage.
 */
template <typename MessageT>
class MonitorController
{
  public:
    /**
     * @brief Construct a new Monitor Controller object
     *
     * @param description : A text label displayed on the left side of the progress bars
     * @param unit : the unit of message count
     * @param determine_count_fn : A function that computes the count for each incoming message
     */
    MonitorController(const std::string& description,
                      std::string unit                                                  = "messages",
                      indicators::Color font_color                                      = indicators::Color::cyan,
                      indicators::FontStyle font_style                                  = indicators::FontStyle::bold,
                      std::optional<std::function<size_t(MessageT)>> determine_count_fn = std::nullopt);

    auto auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>;

    MessageT progress_sink(MessageT msg);
    void sink_on_completed();

  private:
    static std::string format_duration(std::chrono::seconds duration);
    static std::string format_throughput(std::chrono::seconds duration, size_t count, const std::string& unit);

    int m_bar_id;
    const std::string m_unit;
    std::optional<std::function<int(MessageT)>> m_determine_count_fn;
    size_t m_count{0};
    time_point_t m_start_time;
    bool m_is_started{false};
};

template <typename MessageT>
MonitorController<MessageT>::MonitorController(const std::string& description,
                                               std::string unit,
                                               indicators::Color font_color,
                                               indicators::FontStyle font_style,
                                               std::optional<std::function<size_t(MessageT)>> determine_count_fn) :
  m_unit(std::move(unit)),
  m_determine_count_fn(determine_count_fn)
{
    if (!m_determine_count_fn)
    {
        m_determine_count_fn = auto_count_fn();
        if (!m_determine_count_fn)
        {
            throw std::runtime_error("No count function provided and no default count function available");
        }
    }

    m_bar_id = ProgressBarContextManager::get_instance().add_progress_bar(description, font_color, font_style);
}

template <typename MessageT>
MessageT MonitorController<MessageT>::progress_sink(MessageT msg)
{
    if (!m_is_started)
    {
        m_start_time = std::chrono::system_clock::now();
        m_is_started = true;
    }
    m_count += (*m_determine_count_fn)(msg);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - m_start_time);

    auto& manager = ProgressBarContextManager::get_instance();
    auto& pbar    = manager.progress_bars()[m_bar_id];
    pbar->set_option(indicators::option::PostfixText{format_throughput(duration, m_count, m_unit)});
    pbar->tick();

    manager.display();

    return msg;
}

template <typename MessageT>
void MonitorController<MessageT>::sink_on_completed()
{
    auto& manager = ProgressBarContextManager::get_instance();
    auto& pbar    = manager.progress_bars()[m_bar_id];
    pbar->mark_as_completed();
}

template <typename MessageT>
std::string MonitorController<MessageT>::format_duration(std::chrono::seconds duration)
{
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << minutes.count() << "m:" << std::setw(2) << std::setfill('0')
        << seconds.count() << "s";
    return oss.str();
}

template <typename MessageT>
std::string MonitorController<MessageT>::format_throughput(std::chrono::seconds duration,
                                                           size_t count,
                                                           const std::string& unit)
{
    double throughput = static_cast<double>(count) / duration.count();
    std::ostringstream oss;
    oss << count << " " << unit << " in " << format_duration(duration) << ", " << "Throughput: " << std::fixed
        << std::setprecision(2) << throughput << " " << unit << "/s";
    return oss.str();
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

/** @} */  // end of group
}  // namespace morpheus
