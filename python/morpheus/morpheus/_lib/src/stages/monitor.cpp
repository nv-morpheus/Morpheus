#include "morpheus/stages/monitor.hpp"

namespace morpheus {

// Component public implementations
// ****************** MonitorStage ************************ //
template <typename MessageT>
MonitorStage<MessageT>::MonitorStage(const std::string& description,
                                     float smoothing,
                                     const std::string& unit,
                                     bool delayed_start,
                                     std::optional<std::function<int(MessageT)>> determine_count_fn) :
  base_t(base_t::op_factory_from_sub_fn(build_operator()))
{
    m_monitor_controller = MonitorController<MessageT>(description, smoothing, unit, delayed_start, determine_count_fn);
}

template <typename MessageT>
MonitorStage<MessageT>::subscribe_fn_t MonitorStage<MessageT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                m_monitor_controller.progress_sink(msg);
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                m_monitor_controller.sink_on_completed();
                output.on_completed();
            }));
    };
}

// ************ MonitorStageInterfaceProxy ************* //
template <typename MessageT>
std::shared_ptr<mrc::segment::Object<MonitorStage<MessageT>>> MonitorStageInterfaceProxy::init(
    mrc::segment::Builder& builder)
{
    auto stage = builder.construct_object<MonitorStage<MessageT>>();

    return stage;
}
}  // namespace morpheus
