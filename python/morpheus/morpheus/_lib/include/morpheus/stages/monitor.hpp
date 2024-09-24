#pragma once

#include "morpheus/export.h"  // for MORPHEUS_EXPORT

#include <indicators/progress_bar.hpp>
#include <mrc/segment/builder.hpp>  // for Builder
#include <pymrc/node.hpp>           // for PythonNode
#include <rxcpp/rx.hpp>             // for trace_activity, decay_t, from

#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** FilterDetectionStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief
 */
template <typename InputT, typename OutputT>
class MORPHEUS_EXPORT MonitorStage : public mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<InputT>, std::shared_ptr<OutputT>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief
     */
    MonitorStage(const std::string& description, float smoothing, const std::string& unit, bool delayed_start);

  private:
    subscribe_fn_t build_operator();
    const std::string& m_description;
    float m_smoothing;
    const std::string& m_unit;
    bool m_delayed_start;
    long long m_count;

    indicators::ProgressBar m_progress_bar;
};

/****** MonitorStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT MonitorStageInterfaceProxy
{
    template <typename InputT, typename OutputT>
    static std::shared_ptr<mrc::segment::Object<MonitorStage<InputT, OutputT>>> init(mrc::segment::Builder& builder);
};

/** @} */  // end of group
}  // namespace morpheus
