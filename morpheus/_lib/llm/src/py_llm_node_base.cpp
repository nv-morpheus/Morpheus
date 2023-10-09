#include "py_llm_node_base.hpp"

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <mrc/coroutines/task.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymrc/types.hpp>

namespace morpheus::llm {

template <class BaseT>
std::vector<std::string> PyLLMNodeBase<BaseT>::get_input_names() const
{
    MRC_PYBIND11_OVERRIDE_PURE_TEMPLATE(std::vector<std::string>, LLMNodeBase, BaseT, get_input_names);
}

template <class BaseT>
Task<std::shared_ptr<LLMContext>> PyLLMNodeBase<BaseT>::execute(std::shared_ptr<LLMContext> context)
{
    MRC_PYBIND11_OVERRIDE_CORO_PURE_TEMPLATE(std::shared_ptr<LLMContext>, LLMNodeBase, BaseT, execute, context);
}

// explicit instantiations
template class PyLLMNodeBase<>;  // LLMNodeBase
template class PyLLMNodeBase<LLMNode>;
template class PyLLMNodeBase<LLMEngine>;

}  // namespace morpheus::llm
