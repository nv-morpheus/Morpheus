#pragma once

#include <string>
#include <vector>

namespace morpheus::llm {

struct InputMap
{
    std::string input_name;      // The name of the upstream node to use as input
    std::string node_name{"-"};  // The name of the input that the upstream node maps to. '-' is a placeholder for the
                                 // default input of the node
};

// Ordered mapping of input names (current node) to output names (from previous nodes)
using input_map_t = std::vector<InputMap>;

}  // namespace morpheus::llm
