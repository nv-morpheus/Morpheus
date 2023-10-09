#include "py_llm_node.hpp"

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymrc/types.hpp>

namespace morpheus::llm {

template <class BaseT>
std::shared_ptr<LLMNodeRunner> PyLLMNode<BaseT>::add_node(std::string name,
                                                          input_map_t inputs,
                                                          std::shared_ptr<LLMNodeBase> node,
                                                          bool is_output)
{
    // Try to cast the object to a python object to ensure that we keep it alive
    m_py_nodes[node] = pybind11::cast(node);

    // Call the base class implementation
    return LLMNode::add_node(std::move(name), std::move(inputs), std::move(node), is_output);
}

// explicit instantiations
template class PyLLMNode<>;
template class PyLLMNode<LLMEngine>;

}  // namespace morpheus::llm
