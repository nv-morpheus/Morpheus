#include "morpheus/llm/llm_context.hpp"

namespace morpheus::llm {

LLMContext::LLMContext() : m_state(std::make_shared<LLMContextState>())
{
    // m_outputs_future = m_outputs_promise.get_future().share();
}

LLMContext::LLMContext(LLMTask task, std::shared_ptr<ControlMessage> message) : LLMContext()
{
    m_state->task    = std::move(task);
    m_state->message = std::move(message);
}

LLMContext::LLMContext(std::shared_ptr<LLMContext> parent, std::string name, input_map_t inputs) :
  m_parent(std::move(parent)),
  m_name(std::move(name)),
  m_inputs(std::move(inputs))
{
    // m_outputs_future = m_outputs_promise.get_future().share();

    // this->m_parent = parent;
    // this->m_name   = std::move(name);
    // this->m_inputs = std::move(inputs);
}

LLMContext::~LLMContext() = default;

std::shared_ptr<LLMContext> LLMContext::parent() const
{
    return m_parent;
}

const std::string& LLMContext::name() const
{
    return m_name;
}

const input_map_t& LLMContext::input_map() const
{
    return m_inputs;
}

const LLMTask& LLMContext::task() const
{
    if (m_parent)
    {
        return m_parent->task();
    }

    return m_state->task;
}

std::shared_ptr<ControlMessage>& LLMContext::message() const
{
    if (m_parent)
    {
        return m_parent->message();
    }

    return m_state->message;
}

nlohmann::json::const_reference LLMContext::all_outputs() const
{
    return m_outputs;
}

std::string LLMContext::full_name() const
{
    // Determine the full name
    if (m_parent)
    {
        return m_parent->full_name() + "/" + m_name;
    }

    // If we dont have a parent, we are the root context. So return nothing
    return "";
}

std::shared_ptr<LLMContext> LLMContext::push(std::string name, input_map_t inputs)
{
    return std::make_shared<LLMContext>(this->shared_from_this(), std::move(name), std::move(inputs));
}

void LLMContext::pop()
{
    // Copy the outputs from the child context to the parent
    if (m_output_names.empty())
    {
        // Use them all by default
        m_parent->set_output(m_name, std::move(m_outputs));
    }
    else if (m_output_names.size() == 1)
    {
        // Treat only a single output as the output
        m_parent->set_output(m_name, std::move(m_outputs[m_output_names[0]]));
    }
    else
    {
        // Build a new json object with only the specified keys
        nlohmann::json new_outputs;

        for (const auto& output_name : m_output_names)
        {
            new_outputs[output_name] = m_outputs[output_name];
        }

        m_parent->set_output(m_name, std::move(new_outputs));
    }
}

nlohmann::json::const_reference LLMContext::get_input() const
{
    if (m_inputs.size() > 1)
    {
        throw std::runtime_error(
            "LLMContext::get_input() called on a context with multiple inputs. Use get_input(input_name) instead.");
    }

    return this->get_input(m_inputs[0].node_name);

    // nlohmann::json inputs;

    // for (const auto& [input_name, output_name] : m_inputs)
    // {
    //     inputs[input_name] = m_state->outputs[nlohmann::json::json_pointer(output_name)];
    // }

    // return inputs;
}

nlohmann::json::const_reference LLMContext::get_input(const std::string& node_name) const
{
    // if (node_name[0] == '$')
    // {
    //     // Interpolate it as a json path
    //     auto outputs_str = m_outputs.dump();

    //     jsoncons::json tmp_json = jsoncons::json::parse(outputs_str);

    //     std::ostringstream ss;
    //     jsoncons::jsonpath::json_query(tmp_json, node_name).dump_pretty(ss);

    //     LOG(INFO) << ss.str();
    // }

    if (node_name[0] == '/')
    {
        nlohmann::json::json_pointer node_json_ptr(node_name);

        if (!m_outputs.contains(node_json_ptr))
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the output map"));
        }

        // Get the value from a sibling output
        return m_outputs[node_json_ptr];
    }
    else
    {
        // Must be on the parent, so find the mapping between this namespace and the parent
        auto found = std::find_if(m_inputs.begin(), m_inputs.end(), [&node_name](const auto& map_iterator) {
            return map_iterator.node_name == node_name;
        });

        if (found == m_inputs.end())
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the input list"));
        }

        auto& input_name = found->input_name;

        // Get the value from a parent output
        return m_parent->get_input(input_name);
    }
}

nlohmann::json LLMContext::get_inputs() const
{
    nlohmann::json inputs;

    for (const auto& in_map : m_inputs)
    {
        inputs[in_map.node_name] = this->get_input(in_map.node_name);
    }

    return inputs;
}

void LLMContext::set_output(nlohmann::json outputs)
{
    m_outputs = std::move(outputs);
    // auto full_name = nlohmann::json::json_pointer(this->full_name());

    // // if (m_parent)
    // // {
    // //     auto& output = m_parent->get_outputs()
    // // }

    // m_state->values[full_name] = std::move(outputs);

    // // Notify that the outputs are complete
    // this->outputs_complete();
}

void LLMContext::set_output(const std::string& output_name, nlohmann::json outputs)
{
    m_outputs[output_name] = std::move(outputs);
    // std::string full_name = nlohmann::json::json_pointer(this->full_name() + "/" + output_name);

    // m_state->values[full_name] = std::move(outputs);

    // std::vector<int32_t> test(outputs.size(), 0);

    // //           using RepType        = typename ElementTo::rep;
    // //   auto transformer     = fixed_width_type_converter<ElementFrom, RepType>{};
    // //   auto transform_begin = thrust::make_transform_iterator(begin, transformer);
    // //   auto const size      = cudf::distance(begin, end);
    // auto const elements = thrust::host_vector<int32_t>(test.begin(), test.end());
    // auto device_buff =
    //     rmm::device_buffer{elements.data(), test.size() * sizeof(int32_t), cudf::get_default_stream()};

    // // Create a cudf column
    // auto new_column = std::make_unique<cudf::column>(
    //     cudf::data_type{cudf::type_id::INT32}, outputs.size(), std::move(device_buff), rmm::device_buffer{}, 0);

    // m_state->outputs_columns[full_name] = std::move(new_column);
}

void LLMContext::set_output_names(std::vector<std::string> output_names)
{
    m_output_names = std::move(output_names);
}

void LLMContext::outputs_complete()
{
    // m_outputs_promise.set_value();
}

nlohmann::json::const_reference LLMContext::view_outputs() const
{
    // // // Wait for the outputs to be available
    // // m_outputs_future.wait();

    // return m_state->values[this->full_name()];
    return m_outputs;
}

}  // namespace morpheus::llm
