#pragma once

#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace morpheus::llm {

class LLMNode : public LLMNodeBase
{
  public:
    LLMNode();
    ~LLMNode() override;

    virtual std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                                    input_map_t inputs,
                                                    std::shared_ptr<LLMNodeBase> node,
                                                    bool is_output = false);

    std::vector<std::string> get_input_names() const override;

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override;

  private:
    std::vector<std::shared_ptr<LLMNodeRunner>> m_child_runners;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_node_names;  // Names of nodes to be used as the output
};

}  // namespace morpheus::llm
