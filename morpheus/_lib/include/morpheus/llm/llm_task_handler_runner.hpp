#pragma once

#include "morpheus/export.h"
#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_task_handler.hpp"
#include "morpheus/types.hpp"

#include <coroutine>
#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

class MORPHEUS_EXPORT LLMTaskHandlerRunner
{
  public:
    LLMTaskHandlerRunner(input_map_t inputs, std::shared_ptr<LLMTaskHandler> handler);

    ~LLMTaskHandlerRunner();

    virtual Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context);

    const input_map_t& input_names() const
    {
        return m_inputs;
    }

  private:
    input_map_t m_inputs;
    std::shared_ptr<LLMTaskHandler> m_handler;
};

}  // namespace morpheus::llm
