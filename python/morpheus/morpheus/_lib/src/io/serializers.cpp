/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/io/serializers.hpp"

#include "morpheus/objects/data_table.hpp"  // for IDataTable
#include "morpheus/objects/file_types.hpp"
#include "morpheus/objects/table_info_data.hpp"  // for TableInfoData
#include "morpheus/utilities/cudf_util.hpp"

#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>  // for column_name_info, sink_info, table_metadata
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>  // for is_nested
#include <glog/logging.h>
#include <pybind11/gil.h>  // for PyGILState_Check, gil_scoped_acquire
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // for size_t
#include <fstream>
#include <memory>  // for shared_ptr
#include <numeric>
#include <sstream>  // IWYU pragma: keep
#include <vector>
// IWYU pragma: no_include <unordered_map>

namespace py = pybind11;
using namespace py::literals;
using namespace std::string_literals;

namespace {

cudf::io::column_name_info make_column_name_info(std::string name, const py::object& py_col)
{
    // construct a column_name_info from a python column object, loosely based on the _dtype_to_names_list
    // method in cudf's `python/cudf/cudf/io/json.py`
    DCHECK(PyGILState_Check() != 0);
    auto dtypes_mod  = py::module_::import("cudf.core.dtypes");
    auto StructDtype = dtypes_mod.attr("StructDtype");
    auto ListDtype   = dtypes_mod.attr("ListDtype");

    const auto& py_dtype = py_col.attr("dtype");
    bool is_struct_col   = py::isinstance(py_dtype, StructDtype);
    bool is_list_col     = py::isinstance(py_dtype, ListDtype);

    py::list fields;
    if (is_struct_col)
    {
        // Attribute only exists on StructDtype
        fields = py::list(py_col.attr("dtype").attr("fields").attr("keys")());
    }

    std::vector<cudf::io::column_name_info> children;
    if (is_struct_col || is_list_col)
    {
        auto py_children = py_col.attr("children");
        std::size_t i    = 0;

        for (auto& child : py_children)
        {
            // child is a handle
            const auto& py_child = child.cast<py::object>();
            std::string child_name{};
            if (is_struct_col)
            {
                child_name = fields[i].cast<std::string>();
            }

            children.emplace_back(make_column_name_info(child_name, py_child));

            ++i;
        }
    }

    cudf::io::column_name_info col_info{std::move(name)};
    col_info.children = std::move(children);

    return col_info;
}

cudf::io::table_metadata build_cudf_metadata(const cudf::table_view& tbl_view,
                                             const std::vector<std::string>& column_names,
                                             const py::object& df)
{
    std::vector<std::size_t> nested_col_indicies;
    std::vector<cudf::io::column_name_info> column_name_infos(column_names.size());
    for (std::size_t i = 0; i < column_names.size(); ++i)
    {
        if (!cudf::is_nested(tbl_view.column(i).type()))
        {
            column_name_infos[i] = column_names[i];
        }
        else
        {
            nested_col_indicies.push_back(i);
        }
    }

    // If we have a struct column, we need to grab the GIL and inspect the children
    // Remove once https://github.com/rapidsai/cudf/issues/19215 is resolved
    if (!nested_col_indicies.empty())
    {
        pybind11::gil_scoped_acquire gil;

        // we need the column objects not the series objects
        const pybind11::tuple& df_columns = df.attr("_columns");
        const auto num_df_cols            = py::len(df_columns);

        // When the index is included in the output, the index doesn't appear in the DataFrame's _columns
        DCHECK(num_df_cols == column_names.size() || num_df_cols + 1 == column_names.size())
            << "Number of columns in DataFrame does not match number of column names provided";

        const auto col_idx_offset = column_names.size() - num_df_cols;
        for (const auto col_idx : nested_col_indicies)
        {
            const auto& py_col         = df_columns[col_idx - col_idx_offset];
            column_name_infos[col_idx] = make_column_name_info(column_names[col_idx], py_col);
        }
    }

    return cudf::io::table_metadata{std::move(column_name_infos)};
}

}  // namespace

namespace morpheus {

class OStreamSink : public cudf::io::data_sink
{
  public:
    OStreamSink(std::ostream& stream) : m_stream(stream) {}

    /**
     * @brief Append the buffer content to the sink
     *
     * @param[in] data Pointer to the buffer to be written into the sink object
     * @param[in] size Number of bytes to write
     *
     * @return void
     */
    void host_write(void const* data, size_t size) override
    {
        m_stream.write(static_cast<char const*>(data), size);
        m_bytest_written += size;
    }

    /**
     * @brief Flush the data written into the sink
     */
    void flush() override
    {
        m_stream.flush();
    }

    /**
     * @brief Returns the total number of bytes written into this sink
     *
     * @return size_t Total number of bytes written into this sink
     */
    size_t bytes_written() override
    {
        return m_bytest_written;
    }

  private:
    std::ostream& m_stream;
    size_t m_bytest_written{0};
};

void table_to_csv(
    const TableInfoData& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    auto column_names         = tbl.column_names;
    cudf::size_type start_col = 1;
    if (include_index_col)
    {
        start_col = 0;
        column_names.insert(column_names.begin(), ""s);  // insert the id column
    }

    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), start_col);
    auto tbl_view = tbl.table_view.select(col_idexes);

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::csv_writer_options_builder(destination, tbl_view)
                               .include_header(include_header)
                               .true_value("True"s)
                               .false_value("False"s);

    if (include_header)
    {
        options_builder = options_builder.names(column_names);
    }

    cudf::io::write_csv(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_csv(const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    table_to_csv(tbl.get_data(), out_stream, include_header, include_index_col, flush);
}

std::string df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_csv(tbl, out_stream, include_header, include_index_col);

    return out_stream.str();
}

void table_to_json(
    const TableInfoData& tbl, const py::object& df, std::ostream& out_stream, bool include_index_col, bool flush)
{
    if (!include_index_col)
    {
        LOG(WARNING) << "Ignoring include_index_col=false as this isn't supported by cuDF";
    }

    std::vector<cudf::size_type> col_idexes(tbl.column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), 1);
    auto tbl_view = tbl.table_view.select(col_idexes);

    auto tbl_meta = build_cudf_metadata(tbl_view, tbl.column_names, df);

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                               .metadata(std::move(tbl_meta))
                               .lines(true)
                               .include_nulls(true)
                               .na_rep("null");

    cudf::io::write_json(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_json(const TableInfo& tbl, std::ostream& out_stream, bool include_index_col, bool flush)
{
    table_to_json(tbl.get_data(), tbl.get_parent()->get_py_object(), out_stream, include_index_col, flush);
}

std::string df_to_json(const TableInfo& tbl, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_json(tbl, out_stream, include_index_col);

    return out_stream.str();
}

void table_to_parquet(const TableInfoData& tbl,
                      const py::object& df,
                      std::ostream& out_stream,
                      bool include_header,
                      bool include_index_col,
                      bool flush)
{
    auto column_names         = tbl.column_names;
    cudf::size_type start_col = 1;
    if (include_index_col)
    {
        start_col = 0;
        column_names.insert(column_names.begin(), ""s);  // insert the id column
    }

    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), start_col);
    auto tbl_view = tbl.table_view.select(col_idexes);

    cudf::io::table_input_metadata tbl_meta(build_cudf_metadata(tbl_view, column_names, df));

    OStreamSink sink(out_stream);
    auto destination = cudf::io::sink_info(&sink);
    auto options_builder =
        cudf::io::parquet_writer_options_builder(destination, tbl_view).metadata(std::move(tbl_meta));

    cudf::io::write_parquet(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_parquet(
    const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    table_to_parquet(
        tbl.get_data(), tbl.get_parent()->get_py_object(), out_stream, include_header, include_index_col, flush);
}

std::string df_to_parquet(const TableInfo& tbl, bool include_header, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_parquet(tbl, out_stream, include_header, include_index_col);

    return out_stream.str();
}

template <typename T>
T get_with_default(const py::dict& d, const std::string& key, T default_value)
{
    if (d.contains(key))
    {
        return d[key.c_str()].cast<T>();
    }

    return default_value;
}

void SerializersProxy::write_df_to_file(pybind11::object df,
                                        std::string filename,
                                        FileTypes file_type,
                                        const py::kwargs& kwargs)
{
    CudfHelper::load();
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    std::ofstream out_file;
    out_file.open(filename);

    auto tbl = CudfHelper::CudfHelper::table_info_data_from_table(df);

    switch (file_type)
    {
    case FileTypes::JSON: {
        table_to_json(tbl,
                      df,
                      out_file,
                      get_with_default(kwargs, "include_index_col", true),
                      get_with_default(kwargs, "flush", false));
        break;
    }
    case FileTypes::CSV: {
        table_to_csv(tbl,
                     out_file,
                     get_with_default(kwargs, "include_header", true),
                     get_with_default(kwargs, "include_index_col", true),
                     get_with_default(kwargs, "flush", false));
        break;
    }
    case FileTypes::PARQUET: {
        table_to_parquet(tbl,
                         df,
                         out_file,
                         get_with_default(kwargs, "include_header", true),
                         get_with_default(kwargs, "include_index_col", true),
                         get_with_default(kwargs, "flush", false));
        break;
    }
    case FileTypes::Auto:
    default:
        throw std::logic_error(MORPHEUS_CONCAT_STR("Unsupported filetype: " << file_type));
    }
}
}  // namespace morpheus
