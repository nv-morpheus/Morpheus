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
                      float smoothing,
                      const std::string& unit,
                      bool delayed_start,
                      std::optional<std::function<int(MessageT)>> determin_count_fn = std::nullopt);

  private:
    MessageT progress_sink(MessageT msg);
    auto auto_count_fn(MessageT msg) -> std::optional<std::function<int(MessageT)>>;
    void sink_on_completed();

    const std::string& m_description;
    float m_smoothing;
    const std::string& m_unit;
    bool m_delayed_start;
    unsigned long m_count;

    indicators::ProgressBar m_progress_bar;

    std::optional<std::function<int(MessageT)>> m_determine_count_fn;

    static indicators::DynamicProgress<indicators::ProgressBar> m_progress_bars;
};

template <typename InputT>
indicators::DynamicProgress<indicators::ProgressBar> MonitorController<InputT>::m_progress_bars;

/** @} */  // end of group
}  // namespace morpheus
