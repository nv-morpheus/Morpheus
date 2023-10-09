#pragma once

#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_task_handler.hpp"
#include "morpheus/types.hpp"

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace morpheus::llm {

class LLMEngine : public LLMNode
{
  public:
    LLMEngine();
    ~LLMEngine() override;

    virtual void add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler);

    virtual Task<std::vector<std::shared_ptr<ControlMessage>>> run(std::shared_ptr<ControlMessage> input_message);

  private:
    Task<std::vector<std::shared_ptr<ControlMessage>>> handle_tasks(std::shared_ptr<LLMContext> context);

    std::vector<std::shared_ptr<LLMTaskHandlerRunner>> m_task_handlers;
};

}  // namespace morpheus::llm
