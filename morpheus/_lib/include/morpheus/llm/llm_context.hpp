#pragma once

#include "morpheus/export.h"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <mrc/types.hpp>

#include <memory>

namespace morpheus::llm {

struct LLMContextState
{
    LLMTask task;
    std::shared_ptr<ControlMessage> message;
    nlohmann::json values;
};

class MORPHEUS_EXPORT LLMContext : public std::enable_shared_from_this<LLMContext>
{
  public:
    LLMContext();

    LLMContext(LLMTask task, std::shared_ptr<ControlMessage> message);

    LLMContext(std::shared_ptr<LLMContext> parent, std::string name, input_map_t inputs);

    ~LLMContext();

    std::shared_ptr<LLMContext> parent() const;

    const std::string& name() const;

    const input_map_t& input_map() const;

    const LLMTask& task() const;

    std::shared_ptr<ControlMessage>& message() const;

    nlohmann::json::const_reference all_outputs() const;

    std::string full_name() const;

    std::shared_ptr<LLMContext> push(std::string name, input_map_t inputs);

    void pop();

    nlohmann::json::const_reference get_input() const;

    nlohmann::json::const_reference get_input(const std::string& node_name) const;

    nlohmann::json get_inputs() const;

    void set_output(nlohmann::json outputs);

    void set_output(const std::string& output_name, nlohmann::json outputs);

    void set_output_names(std::vector<std::string> output_names);

    void outputs_complete();

    nlohmann::json::const_reference view_outputs() const;

  private:
    std::shared_ptr<LLMContext> m_parent{nullptr};
    std::string m_name;
    input_map_t m_inputs;
    std::vector<std::string> m_output_names;  // Names of keys to be used as the output. Empty means use all keys

    std::shared_ptr<LLMContextState> m_state;

    nlohmann::json m_outputs;

    mrc::Promise<void> m_outputs_promise;
    mrc::SharedFuture<void> m_outputs_future;
};
}  // namespace morpheus::llm
