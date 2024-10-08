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
#include <indicators/dynamic_progress.hpp>
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
  public:
    LineInsertingFilter(size_t num_new_lines) : m_num_new_lines(num_new_lines) {}
    std::string do_filter(const std::string& line)
    {
        std::stringstream new_line;
        // for (size_t i = 0; i < m_num_new_lines; ++i)
        // {
        //     new_line << "\n";
        // }
        // new_line << "\033[" << m_num_new_lines << "A" << "\033[1L" << line;
        // new_line << "\033[1L" << line;
        // return new_line.str();
        return "\n\033[A\033[1L" + line;
    }

  private:
    size_t m_num_new_lines;
};

// See customize_streambuf()
struct LineParsingFilter : boost::iostreams::line_filter
{
  public:
    LineParsingFilter(size_t num_new_lines) : m_num_new_lines(num_new_lines) {}

    std::string do_filter(const std::string& line)
    {
        std::string parsed_line(line);
        size_t found = 0;
        while (found != std::string::npos)
        {
            found = parsed_line.find("\r\r", found);
            if (found != std::string::npos)
            {
                parsed_line.insert(found + 1, "\n");
                found += 6;
            }
        }
        std::stringstream new_line;
        new_line << parsed_line << "\n\033[" << m_num_new_lines << "A";
        return new_line.str();
    }

  private:
    size_t m_num_new_lines;
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

    size_t add_progress_bar(const std::string& description)
    {
        m_progress_bars.push_back(std::move(initialize_progress_bar(description)));
        init_log_buf();
        // init_monitor_buf();

        // DynamicProgress should take ownership over progressbars: https://github.com/p-ranav/indicators/issues/134
        // The fix to this issue is not yet released, so we need to:
        //     - Maintain the lifetime of the progress bar in m_progress_bars
        //     - Push the underlying progress bar object to the DynamicProgress container, since it accepts
        //        Indicator &bar rather than std::unique_ptr<Indicator> bar before the fix
        return m_dynamic_progress_bars.push_back(*m_progress_bars.back());
    }

    // indicators::DynamicProgress<indicators::IndeterminateProgressBar>& dynamic_progress_bars()
    // {
    //     return m_dynamic_progress_bars;
    // }

    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>>& progress_bars()
    {
        return m_progress_bars;
    }

    void display()
    {
        for (auto& pbar : m_progress_bars)
        {
            pbar->print_progress(true);
            m_original_os << "\n";
        }
        m_original_os << "\033[" << m_progress_bars.size() << "A";
    }

    boost::iostreams::filtering_istreambuf& monitoring_ibuf()
    {
        m_monitor_ibuf.reset();
        m_monitor_ibuf.push(LineParsingFilter(m_progress_bars.size()));
        m_monitor_ibuf.push(m_monitor_buf);
        return m_monitor_ibuf;
    }

    std::ostream& original_os()
    {
        return m_original_os;
    }

    bool is_started()
    {
        bool result = m_is_started;
        if (!m_is_started)
        {
            m_is_started = true;
        }
        return result;
    }

    size_t num_progress_bars() const
    {
        return m_progress_bars.size();
    }

  private:
    ProgressBarContextManager() : m_monitor_os(&m_monitor_buf), m_original_os(m_std_out_buf)
    {
        init_log_buf();
    }
    ~ProgressBarContextManager() = default;

    void init_log_buf()
    {
        m_log_buf.reset();
        m_log_buf.push(LineInsertingFilter(m_progress_bars.size()));
        m_log_buf.push(*m_std_out_buf);
        std::cout.rdbuf(&m_log_buf);
    }

    std::unique_ptr<indicators::IndeterminateProgressBar> initialize_progress_bar(const std::string& description)
    {
        auto progress_bar =
            std::make_unique<indicators::IndeterminateProgressBar>(indicators::option::BarWidth{20},
                                                                   indicators::option::Start{"["},
                                                                   indicators::option::Fill{"."},
                                                                   indicators::option::Lead{"o"},
                                                                   indicators::option::End("]"),
                                                                   indicators::option::PrefixText{description},
                                                                   indicators::option::Stream{m_original_os});

        return std::move(progress_bar);
    }

    indicators::DynamicProgress<indicators::IndeterminateProgressBar> m_dynamic_progress_bars;
    std::vector<std::unique_ptr<indicators::IndeterminateProgressBar>> m_progress_bars;

    std::streambuf* m_std_out_buf{std::cout.rdbuf()};
    std::ostream m_original_os;
    boost::iostreams::filtering_ostreambuf m_log_buf;

    std::stringbuf m_monitor_buf;
    std::ostream m_monitor_os;
    boost::iostreams::filtering_istreambuf m_monitor_ibuf;

    bool m_is_started{false};
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
                      std::optional<std::function<size_t(MessageT)>> determine_count_fn = std::nullopt);

    auto auto_count_fn() -> std::optional<std::function<size_t(MessageT)>>;

    MessageT progress_sink(MessageT msg);
    void sink_on_completed();

  private:
    std::unique_ptr<indicators::IndeterminateProgressBar> initialize_progress_bar();

    static std::string format_duration(std::chrono::seconds duration);
    static std::string format_throughput(std::chrono::seconds duration, size_t count, const std::string& unit);

    const std::string& m_description;
    int m_bar_id;
    const std::string m_unit;
    std::optional<std::function<int(MessageT)>> m_determine_count_fn;
    size_t m_count{0};
    time_point_t m_start_time;
    bool m_time_started{false};
    bool is_started{false};
};

template <typename MessageT>
MonitorController<MessageT>::MonitorController(const std::string& description,
                                               std::string unit,
                                               std::optional<std::function<size_t(MessageT)>> determine_count_fn) :
  m_description(description),
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

    m_bar_id = ProgressBarContextManager::get_instance().add_progress_bar(m_description);
}

template <typename MessageT>
MessageT MonitorController<MessageT>::progress_sink(MessageT msg)
{
    if (!m_time_started)
    {
        m_start_time   = std::chrono::system_clock::now();
        m_time_started = true;
    }
    m_count += (*m_determine_count_fn)(msg);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - m_start_time);

    auto& manager       = ProgressBarContextManager::get_instance();
    auto& progress_bars = manager.progress_bars();
    auto& pbar          = progress_bars[m_bar_id];
    pbar->set_option(indicators::option::PostfixText{format_throughput(duration, m_count, m_unit)});
    pbar->tick();

    manager.display();

    return msg;
}

template <typename MessageT>
void MonitorController<MessageT>::sink_on_completed()
{
    // auto& dynamic_progress_bars = ProgressBarContextManager::get_instance().dynamic_progress_bars();
    // dynamic_progress_bars[m_bar_id].mark_as_completed();
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
