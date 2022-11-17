/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "morpheus/messages/meta.hpp"  // for MessageMeta
#include "morpheus/messages/multi.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <pysrf/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, from
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/sink_properties.hpp>    // for SinkProperties<>::sink_type_t
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <memory>
#include <regex>
#include <string>
#include <vector>  // for vector

namespace morpheus {
/****** Component public implementations *******************/
/****** SerializeStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
*/

#pragma GCC visibility push(default)
/**
 * @brief Include & exclude columns from messages. This class filters columns from a `MultiMessage` object emitting a `MessageMeta`.
 */
class SerializeStage : public srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MessageMeta>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Serialize Stage object
     * 
     * @param include : Attributes that are required send to downstream stage. 
     * @param exclude : Attributes that are not required send to downstream stage.
     * @param fixed_columns : When `True` `SerializeStage` will assume that the Dataframe in all messages contain 
     * the same columns as the first message received.
     */
    SerializeStage(const std::vector<std::string> &include,
                   const std::vector<std::string> &exclude,
                   bool fixed_columns = true);

  private:
    void make_regex_objs(const std::vector<std::string> &regex_strs, std::vector<std::regex> &regex_objs);

    bool match_column(const std::vector<std::regex> &patterns, const std::string &column) const;

    bool include_column(const std::string &column) const;

    bool exclude_column(const std::string &column) const;

    TableInfo get_meta(sink_type_t &msg);

    subscribe_fn_t build_operator();

    bool m_fixed_columns;
    std::vector<std::regex> m_include;
    std::vector<std::regex> m_exclude;
    std::vector<std::string> m_column_names;
};

/****** WriteToFileStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct SerializeStageInterfaceProxy
{
    /**
     * @brief Create and initialize a SerializeStage, and return the result
     * 
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param include : Reference to the attributes that are required send to downstream stage. 
     * @param exclude : Reference to the attributes that are not required send to downstream stage.
     * @param fixed_columns : When `True` `SerializeStage` will assume that the Dataframe in all messages contain 
     * the same columns as the first message received.
     * @return std::shared_ptr<srf::segment::Object<SerializeStage>> 
     */
    static std::shared_ptr<srf::segment::Object<SerializeStage>> init(srf::segment::Builder &builder,
                                                                      const std::string &name,
                                                                      const std::vector<std::string> &include,
                                                                      const std::vector<std::string> &exclude,
                                                                      bool fixed_columns = true);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
