/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/serialize.hpp"

#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <exception>
#include <memory>
#include <string>
#include <type_traits>  // for is_same_v
#include <utility>      // for move

// IWYU thinks basic_stringbuf & map are needed for the regex constructor
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <sstream>

namespace morpheus {

constexpr std::regex_constants::syntax_option_type RegexOptions =
    std::regex_constants::ECMAScript | std::regex_constants::icase;

template <typename InputT>
SerializeStage<InputT>::SerializeStage(const std::vector<std::string>& include,
                                       const std::vector<std::string>& exclude,
                                       bool fixed_columns) :
  base_t(base_t::op_factory_from_sub_fn(build_operator())),
  m_fixed_columns{fixed_columns}
{
    make_regex_objs(include, m_include);
    make_regex_objs(exclude, m_exclude);
}

template <typename InputT>
void SerializeStage<InputT>::make_regex_objs(const std::vector<std::string>& regex_strs,
                                             std::vector<std::regex>& regex_objs)
{
    for (const auto& s : regex_strs)
    {
        regex_objs.emplace_back(s, RegexOptions);
    }
}

template <typename InputT>
bool SerializeStage<InputT>::match_column(const std::vector<std::regex>& patterns, const std::string& column) const
{
    for (const auto& re : patterns)
    {
        if (std::regex_match(column, re))
        {
            return true;
        }
    }
    return false;
}

template <typename InputT>
bool SerializeStage<InputT>::include_column(const std::string& column) const
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

template <typename InputT>
bool SerializeStage<InputT>::exclude_column(const std::string& column) const
{
    return match_column(m_exclude, column);
}

template <typename InputT>
std::shared_ptr<SlicedMessageMeta> SerializeStage<InputT>::get_meta(sink_type_t& msg)
{
    // If none of the columns match the include regex patterns or are all are excluded this has the effect
    // of including all of the rows since calling msg->get_meta({}) will return a view with all columns.
    // The Python impl appears to have the same behavior.
    if (!m_fixed_columns || m_column_names.empty())
    {
        m_column_names.clear();

        std::vector<std::string> column_names;

        if constexpr (std::is_same_v<InputT, MultiMessage>)
        {
            column_names = msg->get_meta().get_column_names();
        }
        else
        {
            column_names = msg->payload()->get_info().get_column_names();
        }

        for (const auto& c : column_names)
        {
            if (include_column(c) && !exclude_column(c))
            {
                m_column_names.push_back(c);
            }
        }
    }

    if constexpr (std::is_same_v<InputT, MultiMessage>)
    {
        return std::make_shared<SlicedMessageMeta>(
            msg->meta, msg->mess_offset, msg->mess_offset + msg->mess_count, m_column_names);
    }
    else
    {
        return std::make_shared<SlicedMessageMeta>(msg->payload(), 0, msg->payload()->count(), m_column_names);
    }
}

template <typename InputT>
SerializeStage<InputT>::subscribe_fn_t SerializeStage<InputT>::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t msg) {
                auto next_meta = this->get_meta(msg);

                output.on_next(std::move(next_meta));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

template class SerializeStage<MultiMessage>;
template class SerializeStage<ControlMessage>;

// ************ SerializeStageInterfaceProxy ************* //
std::shared_ptr<mrc::segment::Object<SerializeStageMM>> SerializeStageInterfaceProxy::init_mm(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::vector<std::string>& include,
    const std::vector<std::string>& exclude,
    bool fixed_columns)
{
    auto stage = builder.construct_object<SerializeStageMM>(name, include, exclude, fixed_columns);

    return stage;
}

std::shared_ptr<mrc::segment::Object<SerializeStageCM>> SerializeStageInterfaceProxy::init_cm(
    mrc::segment::Builder& builder,
    const std::string& name,
    const std::vector<std::string>& include,
    const std::vector<std::string>& exclude,
    bool fixed_columns)
{
    auto stage = builder.construct_object<SerializeStageCM>(name, include, exclude, fixed_columns);

    return stage;
}

}  // namespace morpheus
