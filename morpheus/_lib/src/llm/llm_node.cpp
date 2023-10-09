#include "morpheus/llm/llm_node.hpp"

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node_runner.hpp"
#include "morpheus/llm/utils.hpp"

#include <mrc/coroutines/task.hpp>

namespace morpheus::llm {

LLMNode::LLMNode() = default;

LLMNode::~LLMNode() = default;

std::shared_ptr<LLMNodeRunner> LLMNode::add_node(std::string name,
                                                 input_map_t inputs,
                                                 std::shared_ptr<LLMNodeBase> node,
                                                 bool is_output)
{
    // Get the inputs of the current node
    auto input_names = node->get_input_names();

    auto final_inputs = process_input_names(inputs, input_names);

    auto node_runner = std::make_shared<LLMNodeRunner>(std::move(name), std::move(final_inputs), std::move(node));

    // Add the child inputs to the current inputs
    for (const auto& parent_input : node_runner->parent_input_names())
    {
        if (std::find(m_input_names.begin(), m_input_names.end(), parent_input) == m_input_names.end())
        {
            m_input_names.push_back(parent_input);
        }
    }

    // Perform checks that the existing nodes meet the requirements

    m_child_runners.push_back(node_runner);

    if (is_output)
    {
        m_output_node_names.push_back(node_runner->name());
    }

    return node_runner;
}

std::vector<std::string> LLMNode::get_input_names() const
{
    return m_input_names;
}

Task<std::shared_ptr<LLMContext>> LLMNode::execute(std::shared_ptr<LLMContext> context)
{
    for (auto& runner : m_child_runners)
    {
        // Run the child node
        co_await runner->execute(context);

        // Wait for the child node outputs (This will yield if not already available)
        // context->get_outputs();
    }

    // Before returning, set the output names to only propagate the specified outputs
    context->set_output_names(m_output_node_names);

    co_return context;
}

}  // namespace morpheus::llm
