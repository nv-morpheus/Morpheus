#pragma once

#include "morpheus/llm/input_map.hpp"

namespace morpheus::llm {

input_map_t process_input_names(const input_map_t& inputs, const std::vector<std::string>& input_names);

}  // namespace morpheus::llm
