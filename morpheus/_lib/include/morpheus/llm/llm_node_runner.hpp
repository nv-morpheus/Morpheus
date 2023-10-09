#pragma once

#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <coroutine>
#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

class LLMNodeRunner
{
  public:
    LLMNodeRunner(std::string name, input_map_t inputs, std::shared_ptr<LLMNodeBase> node);

    ~LLMNodeRunner();

    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context);

    const std::string& name() const;

    const input_map_t& inputs() const;

    const std::vector<std::string>& sibling_input_names() const;

    const std::vector<std::string>& parent_input_names() const;

  private:
    std::string m_name;
    input_map_t m_inputs;
    std::shared_ptr<LLMNodeBase> m_node;

    std::vector<std::string> m_sibling_input_names;
    std::vector<std::string> m_parent_input_names;
};

}  // namespace morpheus::llm
