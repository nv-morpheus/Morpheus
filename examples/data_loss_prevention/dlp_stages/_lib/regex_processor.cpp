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

#include "regex_processor.h"

#include <cudf/ast/expressions.hpp>  // for cudf::ast::tree, cudf::ast::column_reference, ast_operator
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>           // for make_column_from_scalar
#include <cudf/copying.hpp>                           // for cudf::copy_if_else
#include <cudf/io/types.hpp>                          // for cudf::io::table_metadata and table_with_metadata
#include <cudf/stream_compaction.hpp>                 // for apply_boolean_mask
#include <cudf/strings/combine.hpp>                   // for concatenate
#include <cudf/strings/contains.hpp>                  // for contains_re
#include <cudf/strings/convert/convert_booleans.hpp>  // for from_booleans
#include <cudf/strings/strip.hpp>                     // for strip
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>  // for compute_column
#include <cudf/types.hpp>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>  // for pymrc::import

#include <chrono>
#include <cstddef>
#include <memory>

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
    CHECK(m_regex_patterns.size() > 1) << "C++ impl currently only supports multiple regex patterns";
}

RegexProcessor::subscribe_fn_t RegexProcessor::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t cm_msg) {
                auto time_start       = std::chrono::steady_clock::now();
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

                for (std::size_t i = 1; i < column_references.size(); ++i)
                {
                    if (i == 1)
                    {
                        tree.push(
                            ast::operation{ast::ast_operator::LOGICAL_OR, column_references[0], column_references[1]});
                    }
                    else
                    {
                        tree.push(ast::operation{ast::ast_operator::LOGICAL_OR, tree.back(), column_references[i]});
                    }
                }

                auto boolean_table = cudf::table_view(boolean_column_views);
                const auto& expr   = tree.back();
                auto bool_col      = cudf::compute_column(boolean_table, expr);

                cudf::table_view table_view{table_info.get_view()};

                std::unique_ptr<cudf::column> labels_col;
                if (m_include_pattern_names)
                {
                    auto labels_table = cudf::table(std::move(label_columns));
                    auto labels_view  = labels_table.view();
                    labels_col        = cudf::strings::concatenate(labels_view,
                                                            cudf::string_scalar(", "),
                                                            cudf::string_scalar(""),
                                                            cudf::strings::separator_on_nulls::NO);
                    std::vector<cudf::column_view> columns{table_view.begin(), table_view.end()};
                    columns.push_back(labels_col->view());
                    table_view = cudf::table_view(columns);
                }

                auto table = cudf::apply_boolean_mask(table_view, bool_col->view());

                // Create a table_with_metadata this is copy/pasted from meta.cpp and should probably be a method there
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
                auto new_meta                              = MessageMeta::create_from_cpp(std::move(table_w_meta), 1);
                cm_msg->payload(new_meta);

                auto stop_time = std::chrono::steady_clock::now();
                auto elapsed   = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - time_start).count();
                m_regex_time_ms += elapsed;

                output.on_next(std::move(cm_msg));
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                std::cerr << "Regex stage completed in " << m_regex_time_ms / 1000.0 << " s" << std::endl;
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
        cudf_regex_patterns[i] = cudf::strings::regex_program::create(pattern);
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
