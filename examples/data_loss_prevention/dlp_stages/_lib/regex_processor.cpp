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

#include "regex_processor.hpp"  // IWYU pragma: associated

#include <cudf/ast/expressions.hpp>         // for cudf::ast::tree, cudf::ast::column_reference, ast_operator
#include <cudf/column/column.hpp>           // for cudf::column
#include <cudf/column/column_view.hpp>      // for column_view
#include <cudf/copying.hpp>                 // for cudf::copy_if_else
#include <cudf/io/types.hpp>                // for cudf::io::table_metadata and table_with_metadata
#include <cudf/stream_compaction.hpp>       // for apply_boolean_mask
#include <cudf/strings/combine.hpp>         // for concatenate
#include <cudf/strings/contains.hpp>        // for contains_re
#include <cudf/strings/regex/flags.hpp>     // for capture_groups, regex_flags
#include <cudf/table/table.hpp>             // for table
#include <cudf/table/table_view.hpp>        // for table_view
#include <cudf/transform.hpp>               // for compute_column
#include <glog/logging.h>                   // for CHECK, COMPACT_GOOGLE_LOG_FATAL, LogMessageFatal
#include <morpheus/messages/meta.hpp>       // for MessageMeta
#include <morpheus/objects/table_info.hpp>  // for TableInfo
#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>  // for pymrc::import

#include <cstddef>    // for size_t
#include <exception>  // for exception_ptr
#include <memory>     // for unique_ptr, shared_ptr
#include <ostream>    // for operator<<
#include <utility>    // for move

// IWYU pragma: no_include <unordered_map>
// IWYU pragma: no_include "morpheus/messages/control.hpp"

namespace morpheus_dlp {

RegexProcessor::RegexProcessor(std::string&& source_column_name,
                               std::vector<std::unique_ptr<cudf::strings::regex_program>>&& regex_patterns,
                               std::vector<cudf::string_scalar>&& pattern_name_scalars,
                               bool include_pattern_names) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_source_column_name(std::move(source_column_name)),
  m_regex_patterns(std::move(regex_patterns)),
  m_pattern_name_scalars(std::move(pattern_name_scalars)),
  m_include_pattern_names(include_pattern_names)
{
    CHECK(m_regex_patterns.size() == m_pattern_name_scalars.size())
        << "Number of regex patterns must match number of pattern names";
    CHECK(m_regex_patterns.size() > 0) << "At least one regex pattern must be provided";
}

RegexProcessor::subscribe_fn_t RegexProcessor::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t cm_msg) {
                auto meta             = cm_msg->payload();
                auto table_info       = meta->get_info();
                const auto& col_view  = table_info.get_column(m_source_column_name);
                const auto col_length = col_view.size();

                std::vector<std::unique_ptr<cudf::column>> boolean_columns(m_regex_patterns.size());
                std::vector<cudf::column_view> boolean_column_views(m_regex_patterns.size());
                std::vector<std::unique_ptr<cudf::column>> label_columns;

                namespace ast = cudf::ast;
                ast::tree tree{};
                std::vector<ast::column_reference> column_references;

                for (std::size_t i = 0; i < m_regex_patterns.size(); ++i)
                {
                    // Apply the regex program to the column view
                    boolean_columns[i]      = cudf::strings::contains_re(col_view, *m_regex_patterns[i]);
                    boolean_column_views[i] = boolean_columns[i]->view();

                    column_references.emplace_back(i);
                    tree.push(column_references.back());

                    if (m_include_pattern_names)
                    {
                        label_columns.emplace_back(cudf::copy_if_else(
                            m_pattern_name_scalars[i], cudf::string_scalar("", false), boolean_column_views[i]));
                    }
                }

                // When we have multiple regex patterns, we need to combine the boolean columns into a single boolean
                // column using a logical OR operation.
                std::unique_ptr<cudf::column> bool_col;
                if (boolean_columns.size() > 1)
                {
                    tree.push(
                        ast::operation{ast::ast_operator::LOGICAL_OR, column_references[0], column_references[1]});
                    for (std::size_t i = 2; i < column_references.size(); ++i)
                    {
                        tree.push(ast::operation{ast::ast_operator::LOGICAL_OR, tree.back(), column_references[i]});
                    }

                    auto boolean_table = cudf::table_view(boolean_column_views);
                    const auto& expr   = tree.back();
                    bool_col           = cudf::compute_column(boolean_table, expr);
                }
                else if (boolean_columns.size() == 1)
                {
                    bool_col = std::move(boolean_columns[0]);
                }
                else
                {
                    LOG(FATAL) << "No boolean columns found, this should never happen";
                }

                cudf::table_view table_view{table_info.get_view()};

                std::unique_ptr<cudf::column> labels_col;
                if (m_include_pattern_names)
                {
                    if (label_columns.size() > 1)
                    {
                        auto labels_table = cudf::table(std::move(label_columns));
                        auto labels_view  = labels_table.view();
                        labels_col        = cudf::strings::concatenate(labels_view,
                                                                cudf::string_scalar(", "),
                                                                cudf::string_scalar(""),
                                                                cudf::strings::separator_on_nulls::NO);
                    }
                    else if (label_columns.size() == 1)
                    {
                        labels_col = std::move(label_columns[0]);
                    }
                    else
                    {
                        LOG(FATAL) << "No label columns found, this should never happen";
                    }

                    std::vector<cudf::column_view> columns{table_view.begin(), table_view.end()};
                    columns.push_back(labels_col->view());
                    table_view = cudf::table_view(columns);
                }

                auto table = cudf::apply_boolean_mask(table_view, bool_col->view());

                // If we don't have any rows, we can stop here no need to emit an empty table
                if (table->num_rows() > 0)
                {
                    // Create a table_with_metadata this is copy/pasted from meta.cpp and should probably be a method
                    // there
                    auto column_names = table_info.get_column_names();
                    if (m_include_pattern_names)
                    {
                        column_names.emplace_back("labels");
                    }

                    auto metadata = cudf::io::table_metadata{};

                    metadata.schema_info.reserve(column_names.size() + 1);
                    metadata.schema_info.emplace_back("");

                    for (auto column_name : column_names)
                    {
                        metadata.schema_info.emplace_back(column_name);
                    }

                    cudf::io::table_with_metadata table_w_meta = {std::move(table), std::move(metadata)};
                    auto new_meta = MessageMeta::create_from_cpp(std::move(table_w_meta), 1);
                    cm_msg->payload(new_meta);

                    output.on_next(std::move(cm_msg));
                }
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

std::shared_ptr<mrc::segment::Object<RegexProcessor>> PassThruStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string source_column_name,
    std::map<std::string, std::string> regex_patterns,
    bool include_pattern_names)
{
    std::vector<std::unique_ptr<cudf::strings::regex_program>> cudf_regex_patterns(regex_patterns.size());
    std::vector<cudf::string_scalar> pattern_name_scalars;

    std::size_t i = 0;
    for (auto& [pattern_name, pattern] : regex_patterns)
    {
        cudf_regex_patterns[i] = cudf::strings::regex_program::create(
            pattern, cudf::strings::regex_flags::DEFAULT, cudf::strings::capture_groups::NON_CAPTURE);
        pattern_name_scalars.emplace_back(pattern_name);
        ++i;
    }
    return builder.construct_object<RegexProcessor>(name,
                                                    std::move(source_column_name),
                                                    std::move(cudf_regex_patterns),
                                                    std::move(pattern_name_scalars),
                                                    include_pattern_names);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(regex_processor, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<RegexProcessor>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<RegexProcessor>>>(m, "RegexProcessor", py::multiple_inheritance())
        .def(py::init<>(&PassThruStageInterfaceProxy::init),
             py::arg("builder"),
             py::arg("name"),
             py::arg("source_column_name"),
             py::arg("regex_patterns"),
             py::arg("include_pattern_names"));
}

}  // namespace morpheus_dlp
