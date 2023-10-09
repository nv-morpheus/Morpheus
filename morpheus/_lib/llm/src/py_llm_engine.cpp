#include "py_llm_engine.hpp"

#include "py_llm_task_handler.hpp"

namespace morpheus::llm {

PyLLMEngine::PyLLMEngine() : PyLLMNode<LLMEngine>() {}

PyLLMEngine::~PyLLMEngine() = default;

void PyLLMEngine::add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler)
{
    // Try to cast the object to a python object to ensure that we keep it alive
    auto py_task_handler = std::dynamic_pointer_cast<PyLLMTaskHandler>(task_handler);

    if (py_task_handler)
    {
        // Store the python object to keep it alive
        m_py_task_handler[task_handler] = pybind11::cast(task_handler);
    }

    // Call the base class implementation
    LLMEngine::add_task_handler(std::move(inputs), std::move(task_handler));
}

}  // namespace morpheus::llm
