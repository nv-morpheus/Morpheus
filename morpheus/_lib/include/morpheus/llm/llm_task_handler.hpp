#pragma once

#include "morpheus/llm/fwd.hpp"
#include "morpheus/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morpheus::llm {

class LLMTaskHandler
{
  public:
    using return_t = std::optional<std::vector<std::shared_ptr<ControlMessage>>>;

    virtual ~LLMTaskHandler() = default;

    virtual std::vector<std::string> get_input_names() const               = 0;
    virtual Task<return_t> try_handle(std::shared_ptr<LLMContext> context) = 0;
};

}  // namespace morpheus::llm
