#include "morpheus/llm/llm_task.hpp"

namespace morpheus::llm {

LLMTask::LLMTask() = default;

LLMTask::LLMTask(std::string task_type, nlohmann::json task_dict) :
  task_type(std::move(task_type)),
  task_dict(std::move(task_dict))
{}

LLMTask::~LLMTask() = default;

size_t LLMTask::size() const
{
    return this->task_dict.size();
}

const nlohmann::json& LLMTask::get(const std::string& key) const
{
    return this->task_dict.at(key);
}

void LLMTask::set(const std::string& key, nlohmann::json&& value)
{
    this->task_dict[key] = std::move(value);
}

}  // namespace morpheus::llm
