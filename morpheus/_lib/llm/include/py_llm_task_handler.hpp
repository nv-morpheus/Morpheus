#pragma once

#include "morpheus/llm/llm_task_handler.hpp"

#include <pybind11/pybind11.h>

namespace morpheus::llm {

class PyLLMTaskHandler : public LLMTaskHandler
{
  public:
    using LLMTaskHandler::LLMTaskHandler;

    ~PyLLMTaskHandler() override;

    std::vector<std::string> get_input_names() const override;

    Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context) override;
};

}  // namespace morpheus::llm
