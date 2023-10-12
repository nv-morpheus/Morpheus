/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/llm/utils.hpp"

#include "morpheus/utilities/string_util.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace morpheus::llm {

// Use this regex
const std::regex VALID_INPUT_NAME(R"([a-zA-Z_][a-zA-Z0-9_]*)", std::regex_constants::ECMAScript);

bool is_valid_node_name(std::string_view name)
{
    return std::regex_match(name.begin(), name.end(), VALID_INPUT_NAME);
}

input_mapping_t process_input_names(const input_mapping_t& inputs, const std::vector<std::string>& input_names)
{
    input_mapping_t final_inputs;
    input_mapping_t placeholder_inputs;

    // Perform any placeholder replacements
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const auto& single_input = inputs[i];

        bool found_star_input_name = single_input.external_name.find('*') != std::string::npos;
        bool found_star_node_name  = single_input.internal_name == "*";

        if (found_star_input_name != found_star_node_name)
        {
            throw std::runtime_error(
                "LLMNode::add_node() called with a placeholder input name and node name that "
                "do not match");
        }
        else if (found_star_input_name && found_star_node_name)
        {
            // Need to process these after the non-placeholder inputs
            placeholder_inputs.push_back(single_input);
        }
        else
        {
            // No placeholder, so just add the input. If the node_name == "-", then replace it with the input name
            if (single_input.internal_name == "-")
            {
                // If we start with a slash, that means we are mapping from another node, not a parent.
                if (single_input.external_name[0] == '/')
                {
                    if (inputs.size() != input_names.size())
                    {
                        throw std::runtime_error(MORPHEUS_CONCAT_STR(
                            "When mapping from a sibling node, the number of siblings must match. Provided: "
                            << inputs.size() << ", Expected: " << input_names.size()));
                    }

                    // Match by index
                    final_inputs.push_back({single_input.external_name, input_names[i]});
                }
                else
                {
                    // Match by name
                    auto found = std::find(input_names.begin(), input_names.end(), single_input.external_name);

                    if (found != input_names.end())
                    {
                        final_inputs.push_back({single_input.external_name, *found});
                    }
                    else if (input_names.size() == 1)
                    {
                        // We can infer that the node name is the only one
                        final_inputs.push_back({single_input.external_name, input_names[0]});
                    }
                    else
                    {
                        throw std::runtime_error(MORPHEUS_CONCAT_STR("Could not find a matching node name for input '"
                                                                     << single_input.external_name << "'"));
                    }
                }
            }
            else
            {
                final_inputs.push_back(single_input);
            }
        }
    }

    if (!placeholder_inputs.empty())
    {
        // TODO(MDD): Support multiple placeholders
        CHECK_EQ(placeholder_inputs.size(), 1) << "Only a single placeholder input is currently supported";

        std::set<std::string> specified_names;

        std::transform(final_inputs.begin(),
                       final_inputs.end(),
                       std::inserter(specified_names, specified_names.begin()),
                       [](const auto& input) {
                           return input.internal_name;
                       });

        std::set<std::string> total_names(input_names.begin(), input_names.end());

        std::vector<std::string> remaining_names;

        // Find the remaining names
        std::set_difference(total_names.begin(),
                            total_names.end(),
                            specified_names.begin(),
                            specified_names.end(),
                            std::back_inserter(remaining_names));

        auto star_input_name_loc = placeholder_inputs[0].external_name.find('*');

        // Loop over the remaining names and add them to the final inputs
        for (const auto& remaining_name : remaining_names)
        {
            // Make a copy of the string to avoid modifying the original
            auto replaced = std::string(placeholder_inputs[0].external_name);
            replaced.replace(star_input_name_loc, 1, remaining_name);
            final_inputs.push_back({replaced, remaining_name});
        }
    }

    if (input_names.size() != final_inputs.size())
    {
        throw std::runtime_error(MORPHEUS_CONCAT_STR(
            "The number of inputs provided does not match the number of inputs expected by the node. Provided: "
            << final_inputs.size() << ", Expected: " << input_names.size()));
    }

    return final_inputs;
}

}  // namespace morpheus::llm
