#include "py_llm_node_base.hpp"

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymrc/types.hpp>

namespace morpheus::llm {

template <class BaseT>
std::vector<std::string> PyLLMNodeBase<BaseT>::get_input_names() const
{
    // Call the python overridden function
    pybind11::gil_scoped_acquire gil;
    pybind11::function override = pybind11::get_override(static_cast<const BaseT*>(this), "get_input_names");

    if (override)
    {
        auto o = override();
        if (pybind11::detail::cast_is_temporary_value_reference<std::vector<std::string>>::value)
        {
            static pybind11::detail::override_caster_t<std::vector<std::string>> caster;
            return pybind11::detail::cast_ref<std::vector<std::string>>(std::move(o), caster);
        }
        return pybind11::detail::cast_safe<std::vector<std::string>>(std::move(o));
    }

    if constexpr (std::is_same_v<BaseT, LLMNodeBase>)
    {
        // Cant call the base class implementation on abstract class
        pybind11::pybind11_fail(
            "Tried to call pure virtual function \""
            "LLMNodeBase"
            "::"
            "get_input_names"
            "\"");
    }
    else
    {
        return BaseT::get_input_names();
    }
}

template <class BaseT>
Task<std::shared_ptr<LLMContext>> PyLLMNodeBase<BaseT>::execute(std::shared_ptr<LLMContext> context)
{
    using return_t = std::shared_ptr<LLMContext>;

    pybind11::gil_scoped_acquire gil;

    pybind11::function override = pybind11::get_override(static_cast<const BaseT*>(this), "execute");

    if (override)
    {
        auto override_coro = override(context);

        // Now determine if the override result is a coroutine or not
        if (!pybind11::module::import("asyncio").attr("iscoroutine")(override_coro).cast<bool>())
        {
            throw std::runtime_error("Must return a coroutine");
        }

        auto override_task = pybind11::module::import("asyncio").attr("create_task")(override_coro);

        mrc::pymrc::PyHolder override_result;
        {
            // Release the GIL before awaiting
            pybind11::gil_scoped_release nogil;

            override_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(override_task));
        }

        // Now cast back to the C++ type
        if (pybind11::detail::cast_is_temporary_value_reference<return_t>::value)
        {
            static pybind11::detail::override_caster_t<return_t> caster;
            co_return pybind11::detail::cast_ref<return_t>(std::move(override_result), caster);
        }
        co_return pybind11::detail::cast_safe<return_t>(std::move(override_result));
    }

    if constexpr (std::is_same_v<BaseT, LLMNodeBase>)
    {
        // Cant call the base class implementation on abstract class
        pybind11::pybind11_fail(
            "Tried to call pure virtual function \""
            "LLMNodeBase"
            "::"
            "execute"
            "\"");
    }
    else
    {
        co_return co_await BaseT::execute(context);
    }
}

// explicit instantiations
template class PyLLMNodeBase<>;  // LLMNodeBase

}  // namespace morpheus::llm
