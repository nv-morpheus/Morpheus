#include "morpheus/stages/monitor.hpp"
#include "indicators/setting.hpp"

namespace morpheus {

// Component public implementations
// ****************** MonitorStage ************************ //
template <typename InputT, typename OutputT>
MonitorStage<InputT, OutputT>::MonitorStage(const std::string& description,
                                            float smoothing,
                                            const std::string& unit,
                                            bool delayed_start) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_description(description),
  m_smoothing(smoothing),
  m_unit(unit),
  m_delayed_start(delayed_start),
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
    // m_progress_bar.set_option(indicators::option::ShowRemainingTime{true});
}

template <typename InputT, typename OutputT>
MonitorStage<InputT, OutputT>::subscribe_fn_t MonitorStage<InputT, OutputT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                m_count++;
                m_progress_bar.set_progress(m_count);
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

// ************ MonitorStageInterfaceProxy ************* //
template <typename InputT, typename OutputT>
std::shared_ptr<mrc::segment::Object<MonitorStage<InputT, OutputT>>> MonitorStageInterfaceProxy::init(
    mrc::segment::Builder& builder)
{
    auto stage = builder.construct_object<MonitorStage<InputT, OutputT>>();

    return stage;
}
}  // namespace morpheus
