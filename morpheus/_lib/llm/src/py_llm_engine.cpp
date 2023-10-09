#include "py_llm_engine.hpp"

#include "py_llm_task_handler.hpp"

namespace morpheus::llm {

PyLLMEngine::PyLLMEngine() : PyLLMNode<LLMEngine>()
{
    // std::promise<void> loop_ready;

    // auto future = loop_ready.get_future();

    // auto setup_debugging = create_gil_initializer();

    // m_thread = std::thread(
    //     [this](std::promise<void> loop_ready, std::function<void()> setup_debugging) {
    //         // Acquire the GIL (and also initialize the ThreadState)
    //         py::gil_scoped_acquire acquire;

    //         // Initialize the debugger
    //         setup_debugging();

    //         py::print("Creating loop");

    //         // Gets (or more likely, creates) an event loop and runs it forever until stop is called
    //         m_loop = py::module::import("asyncio").attr("new_event_loop")();

    //         py::print("Setting loop current");

    //         // Set the event loop as the current event loop
    //         py::module::import("asyncio").attr("set_event_loop")(m_loop);

    //         py::print("Signaling promise");

    //         // Signal we are ready
    //         loop_ready.set_value();

    //         py::print("Running forever");

    //         m_loop.attr("run_forever")();
    //     },
    //     std::move(loop_ready),
    //     std::move(setup_debugging));

    // py::print("Waiting for startup");
    // {
    //     // Free the GIL otherwise we deadlock
    //     py::gil_scoped_release nogil;

    //     future.get();
    // }

    // // Finally, try and see if our LLM Service is a python object and keep it alive
    // auto py_llm_service = std::dynamic_pointer_cast<PyLLMService>(llm_service);

    // if (py_llm_service)
    // {
    //     // Store the python object to keep it alive
    //     m_py_llm_service = py::cast(llm_service);

    //     // Also, set the loop on the service
    //     py_llm_service->set_loop(m_loop);
    // }

    // py::print("Engine started");
}

PyLLMEngine::~PyLLMEngine() = default;

void PyLLMEngine::add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler)
{
    // Try to cast the object to a python object to ensure that we keep it alive
    auto py_task_handler = std::dynamic_pointer_cast<PyLLMTaskHandler>(task_handler);

    if (py_task_handler)
    {
        // Store the python object to keep it alive
        m_py_task_handler[task_handler] = pybind11::cast(task_handler);

        // // Also, set the loop on the service
        // py_task_handler->set_loop(m_loop);
    }

    // Call the base class implementation
    LLMEngine::add_task_handler(std::move(inputs), std::move(task_handler));
}

}  // namespace morpheus::llm
