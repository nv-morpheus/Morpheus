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
    using return_t = LLMTaskHandler::return_t;

    pybind11 ::gil_scoped_acquire gil;

    pybind11 ::function override = pybind11 ::get_override(static_cast<const LLMTaskHandler*>(this), "try_handle");

    if (!override)
    {
        // Problem
        pybind11 ::pybind11_fail(
            "Tried to call pure virtual function \""
            "LLMTaskHandler"
            "::"
            "try_handle"
            "\"");
    }

    auto override_coro = override(context);

    // Now determine if the override result is a coroutine or not
    if (!py::module::import("asyncio").attr("iscoroutine")(override_coro).cast<bool>())
    {
        throw std::runtime_error("Must return a coroutine");
    }

    auto override_task = py::module::import("asyncio").attr("create_task")(override_coro);

    mrc::pymrc::PyHolder override_result;
    {
        // Release the GIL before awaiting
        pybind11::gil_scoped_release nogil;

        override_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(override_task));
    }

    // Now cast back to the C++ type
    if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
    {
        static pybind11 ::detail ::override_caster_t<return_t> caster;
        co_return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
    }
    co_return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
}

}  // namespace morpheus::llm
