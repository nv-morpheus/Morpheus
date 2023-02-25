/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/types.hpp>  // for column_name_info, sink_info, table_metadata
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/mr/device/per_device_resource.hpp>

#include <array>    // for array
#include <cstddef>  // for size_t
#include <numeric>
#include <ostream>
#include <sstream>  // IWYU pragma: keep
#include <utility>
#include <vector>
// IWYU pragma: no_include <unordered_map>

namespace morpheus {
namespace py = pybind11;
using namespace py::literals;
using namespace std::string_literals;

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

std::string df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_csv(tbl, out_stream, include_header, include_index_col);

    return out_stream.str();
}

void df_to_csv(const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    auto column_names         = tbl.get_column_names();
    cudf::size_type start_col = 1;
    if (include_index_col)
    {
        start_col = 0;
        column_names.insert(column_names.begin(), ""s);  // insert the id column
    }

    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), start_col);
    auto tbl_view = tbl.get_view().select(col_idexes);

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::csv_writer_options_builder(destination, tbl_view)
                               .include_header(include_header)
                               .true_value("True"s)
                               .false_value("False"s);

    cudf::io::table_metadata metadata{};

    if (include_header)
    {
        metadata.column_names = column_names;

        // After cuDF PR #11364, use schema_info instead of column_names (actually just set both)
        metadata.schema_info = std::vector<cudf::io::column_name_info>();

        for (auto& name : column_names)
        {
            metadata.schema_info.emplace_back(cudf::io::column_name_info{name});
        }

        options_builder = options_builder.metadata(&metadata);
    }

    cudf::io::write_csv(options_builder.build(), rmm::mr::get_current_device_resource());

    if (flush)
    {
        sink.flush();
    }
}

std::string df_to_json(MutableTableInfo& tbl, bool include_index_col)
{
    std::string results;
    // no cpp impl for to_json, instead python module converts to pandas and calls to_json
    {
        py::gil_scoped_acquire gil;
        py::object StringIO = py::module_::import("io").attr("StringIO");

        auto df = tbl.checkout_obj();

        auto buffer     = StringIO();
        py::dict kwargs = py::dict("orient"_a = "records", "lines"_a = true, "index"_a = include_index_col);
        df.attr("to_json")(buffer, **kwargs);
        buffer.attr("seek")(0);

        tbl.return_obj(std::move(df));

        py::object pyresults = buffer.attr("getvalue")();
        results              = pyresults.cast<std::string>();
    }

    return results;
}

void df_to_json(MutableTableInfo& tbl, std::ostream& out_stream, bool include_index_col, bool flush)
{
    // Unlike df_to_csv, we use the ostream overload to call the string overload because there is no C++
    // implementation of to_json
    std::string output = df_to_json(tbl, include_index_col);

    // Now write the contents to the stream
    out_stream.write(output.data(), output.size());

    if (flush)
    {
        out_stream.flush();
    }
}

}  // namespace morpheus
