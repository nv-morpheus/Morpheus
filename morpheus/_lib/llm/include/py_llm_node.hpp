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

  private:
    std::map<std::shared_ptr<LLMNodeBase>, pybind11::object> m_py_nodes;
};

}  // namespace morpheus::llm
