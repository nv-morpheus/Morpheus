#include "py_llm_task_handler.hpp"

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/llm_context.hpp"

#include <mrc/coroutines/task.hpp>
#include <pymrc/types.hpp>

namespace morpheus::llm {
namespace py = pybind11;

PyLLMTaskHandler::~PyLLMTaskHandler() = default;

std::vector<std::string> PyLLMTaskHandler::get_input_names() const
{
    PYBIND11_OVERRIDE_PURE(std::vector<std::string>, LLMTaskHandler, get_input_names);
}

Task<LLMTaskHandler::return_t> PyLLMTaskHandler::try_handle(std::shared_ptr<LLMContext> context)
{
    MRC_PYBIND11_OVERRIDE_CORO_PURE(LLMTaskHandler::return_t, LLMTaskHandler, try_handle, context);
}

}  // namespace morpheus::llm
