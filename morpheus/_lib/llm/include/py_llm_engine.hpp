#pragma once

#include "py_llm_node.hpp"

#include "morpheus/llm/llm_engine.hpp"

namespace morpheus::llm {

class PyLLMEngine : public PyLLMNode<LLMEngine>
{
  public:
    PyLLMEngine();

    ~PyLLMEngine() override;

    // ~PyLLMEngine()
    // {
    //     // Acquire the GIL on this thread and call stop on the event loop
    //     py::gil_scoped_acquire acquire;

    //     m_loop.attr("stop")();

    //     // Finally, join on the thread
    //     m_thread.join();
    // }

    void add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler) override;

    // const py::object& get_loop() const
    // {
    //     return m_loop;
    // }

    // std::vector<std::shared_ptr<ControlMessage>> run(std::shared_ptr<ControlMessage> input_message) override
    // {
    //     std::vector<std::shared_ptr<ControlMessage>> output_messages;

    //     return output_messages;
    // }

  private:
    // std::thread m_thread;
    // py::object m_loop;

    // Keep the python objects alive by saving references in this object
    // py::object m_py_llm_service;
    std::map<std::shared_ptr<LLMTaskHandler>, pybind11::object> m_py_task_handler;
};

}  // namespace morpheus::llm
