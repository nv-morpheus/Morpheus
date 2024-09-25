#include "../test_utils/common.hpp"  // for get_morpheus_root, TEST_CLASS_WITH_PYTHON, morpheus

#include "morpheus/controllers/monitor_controller.hpp"  // for MonitorController
#include "morpheus/stages/monitor.hpp"                  // for MonitorStage

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <numeric>
#include <vector>

using namespace morpheus;

TEST_CLASS(MonitorController);

std::unique_ptr<cudf::table> create_cudf_table(int rows, int cols)
{
    std::vector<std::unique_ptr<cudf::column>> columns;

    for (int i = 0; i < cols; ++i)
    {
        // Create a numeric column of type INT32 with 'rows' elements
        auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, rows);
        auto col_view = col->mutable_view();

        // Fill the column with range [0, rows - 1]
        std::vector<int32_t> data(rows);
        std::iota(data.begin(), data.end(), 0);
        cudaMemcpy(col_view.data<int32_t>(), data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

        // Add the column to the vector
        columns.push_back(std::move(col));
    }

    // Create and return the table
    return std::make_unique<cudf::table>(std::move(columns));
}

TEST_F(TestMonitorController, TestAutoCountFn)
{
    MonitorController<int> a("test");
}
