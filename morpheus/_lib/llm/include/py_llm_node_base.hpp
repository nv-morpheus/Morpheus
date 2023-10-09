#pragma once

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/fwd.hpp"
#include "morpheus/types.hpp"

#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

namespace morpheus::llm {

template <class BaseT = LLMNodeBase>
class PyLLMNodeBase : public BaseT
{
  public:
    using BaseT::BaseT;

    std::vector<std::string> get_input_names() const override;

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override;

  private:
    std::map<std::shared_ptr<LLMNodeBase>, pybind11::object> m_py_nodes;
};

}  // namespace morpheus::llm
