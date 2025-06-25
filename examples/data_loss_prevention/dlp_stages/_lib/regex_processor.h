/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/scalar/scalar.hpp>                // for cudf::string_scalar
#include <cudf/strings/regex/regex_program.hpp>  // for cudf::strings::regex_program
#include <morpheus/export.h>                     // for exporting symbols
#include <morpheus/messages/control.hpp>         // for ControlMessage
#include <mrc/segment/builder.hpp>               // for Segment Builder
#include <mrc/segment/object.hpp>                // for Segment Object
#include <pymrc/node.hpp>                        // for PythonNode
#include <rxcpp/rx.hpp>

#include <map>     // for map
#include <memory>  // for shared_ptr
#include <string>  // for string

namespace morpheus_dlp {

using namespace morpheus;

// pybind11 sets visibility to hidden by default; we want to export our symbols
class MORPHEUS_EXPORT RegexProcessor
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    RegexProcessor(std::string&& source_column_name,
                   std::vector<std::unique_ptr<cudf::strings::regex_program>>&& regex_patterns,
                   std::vector<cudf::string_scalar>&& pattern_name_scalars,
                   bool include_pattern_names);

    subscribe_fn_t build_operator();

  private:
    std::string m_source_column_name;
    std::vector<std::unique_ptr<cudf::strings::regex_program>> m_regex_patterns;
    std::vector<cudf::string_scalar> m_pattern_name_scalars;
    long m_regex_time_ms         = 0;
    bool m_include_pattern_names = false;
};

struct MORPHEUS_EXPORT PassThruStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<RegexProcessor>> init(mrc::segment::Builder& builder,
                                                                      const std::string& name,
                                                                      std::string source_column_name,
                                                                      std::map<std::string, std::string> regex_patterns,
                                                                      bool include_pattern_names);
};

}  // namespace morpheus_dlp
