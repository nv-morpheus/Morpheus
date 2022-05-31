/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/stages/serialize.hpp>

#include <exception>
#include <memory>
#include <string>

namespace morpheus {

using namespace std::literals;

constexpr std::regex_constants::syntax_option_type RegexOptions =
    std::regex_constants::ECMAScript | std::regex_constants::icase;

// Component public implementations
// ************ WriteToFileStage **************************** //
SerializeStage::SerializeStage(const std::vector<std::string> &include,
                               const std::vector<std::string> &exclude,
                               bool fixed_columns) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_fixed_columns{fixed_columns}
{
    make_regex_objs(include, m_include);
    make_regex_objs(exclude, m_exclude);
}

void SerializeStage::make_regex_objs(const std::vector<std::string> &regex_strs, std::vector<std::regex> &regex_objs)
{
    for (const auto &s : regex_strs)
    {
        regex_objs.emplace_back(std::regex{s, RegexOptions});
    }
}

bool SerializeStage::match_column(const std::vector<std::regex> &patterns, const std::string &column) const
{
    for (const auto &re : patterns)
    {
        if (std::regex_match(column, re))
        {
            return true;
        }
    }
    return false;
}

bool SerializeStage::include_column(const std::string &column) const
{
    if (m_include.empty())
    {
        return true;
    }
    else
    {
        return match_column(m_include, column);
    }
}

bool SerializeStage::exclude_column(const std::string &column) const
{
    return match_column(m_exclude, column);
}

TableInfo SerializeStage::get_meta(sink_type_t &msg)
{
    // If none of the columns match the include regex patterns or are all are excluded this has the effect
    // of including all of the rows since calling msg->get_meta({}) will return a view with all columns.
    // The Python impl appears to have the same behavior.
    if (!m_fixed_columns || m_column_names.empty())
    {
        m_column_names.clear();
        for (const auto &c : msg->get_meta().get_column_names())
        {
            if (include_column(c) && !exclude_column(c))
            {
                m_column_names.push_back(c);
            }
        }
    }

    return msg->get_meta(m_column_names);
}

SerializeStage::subscribe_fn_t SerializeStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                auto table_info = this->get_meta(msg);
                std::shared_ptr<MessageMeta> meta;
                {
                    pybind11::gil_scoped_acquire gil;
                    meta = MessageMeta::create_from_python(std::move(table_info.as_py_object()));
                }
                output.on_next(std::move(meta));
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ WriteToFileStageInterfaceProxy ************* //
std::shared_ptr<neo::segment::Object<SerializeStage>> SerializeStageInterfaceProxy::init(
    neo::segment::Builder &parent,
    const std::string &name,
    const std::vector<std::string> &include,
    const std::vector<std::string> &exclude,
    bool fixed_columns)
{
    auto stage = parent.construct_object<SerializeStage>(name, include, exclude, fixed_columns);

    return stage;
}
}  // namespace morpheus
