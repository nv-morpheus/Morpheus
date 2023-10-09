#pragma once

#include "morpheus/export.h"
#include "morpheus/llm/fwd.hpp"
#include "morpheus/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

class MORPHEUS_EXPORT LLMNodeBase
{
  public:
    virtual ~LLMNodeBase() = default;

    virtual std::vector<std::string> get_input_names() const                               = 0;
    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) = 0;
};

}  // namespace morpheus::llm
