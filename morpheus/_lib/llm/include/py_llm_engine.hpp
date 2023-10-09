#pragma once

#include "py_llm_node.hpp"

#include "morpheus/llm/llm_engine.hpp"

namespace morpheus::llm {

class PyLLMEngine : public PyLLMNode<LLMEngine>
{
  public:
    PyLLMEngine();

    ~PyLLMEngine() override;

    void add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler) override;

  private:
    // Keep the python objects alive by saving references in this object
    std::map<std::shared_ptr<LLMTaskHandler>, pybind11::object> m_py_task_handler;
};

}  // namespace morpheus::llm
