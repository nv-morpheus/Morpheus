#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace morpheus::llm {

struct LLMTask
{
    LLMTask();
    LLMTask(std::string task_type, nlohmann::json task_dict);

    ~LLMTask();

    std::string task_type;

    size_t size() const;

    const nlohmann::json& get(const std::string& key) const;

    void set(const std::string& key, nlohmann::json&& value);

    nlohmann::json task_dict;
};

}  // namespace morpheus::llm
