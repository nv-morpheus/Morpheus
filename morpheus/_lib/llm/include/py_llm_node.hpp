#pragma once

#include "py_llm_node_base.hpp"

#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/input_map.hpp"

#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

namespace morpheus::llm {

template <class BaseT = LLMNode>
class PyLLMNode : public PyLLMNodeBase<BaseT>
{
  public:
    using PyLLMNodeBase<BaseT>::PyLLMNodeBase;

    std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                            input_map_t inputs,
                                            std::shared_ptr<LLMNodeBase> node,
                                            bool is_output = false) override;

    // void execute(std::shared_ptr<LLMContext> context) override
    // {
    //     pybind11 ::gil_scoped_acquire gil;

    //     pybind11 ::function override = pybind11 ::get_override(static_cast<const BaseT*>(this), "execute");

    //     if (!override)
    //     {
    //         // Problem
    //         pybind11 ::pybind11_fail(
    //             "Tried to call pure virtual function \""
    //             "LLMNodeBase"
    //             "::"
    //             "execute"
    //             "\"");
    //     }

    //     auto override_result = override(context);

    //     // Now determine if the override result is a coroutine or not
    //     if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
    //     {
    //         py::print("Returned a coroutine");

    //         auto loop = py::module::import("asyncio").attr("get_running_loop")();

    //         // Need to schedule the result to run on the loop
    //         auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, loop);

    //         // We are a dask future. Quickly check if its done, then release
    //         while (!future.attr("done")().cast<bool>())
    //         {
    //             // Release the GIL and wait for it to be done
    //             py::gil_scoped_release nogil;

    //             boost::this_fiber::yield();
    //         }

    //         // // Completed, move into the returned object
    //         // override_result = future.attr("result")();
    //     }
    //     else
    //     {
    //         py::print("Did not return a coroutine");
    //     }

    //     // // Now cast back to the C++ type
    //     // if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
    //     // {
    //     //     static pybind11 ::detail ::override_caster_t<return_t> caster;
    //     //     return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
    //     // }
    //     // return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
    // }

  private:
    std::map<std::shared_ptr<LLMNodeBase>, pybind11::object> m_py_nodes;
};

}  // namespace morpheus::llm
