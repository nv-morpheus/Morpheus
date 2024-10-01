#include "../test_utils/common.hpp"  // for get_morpheus_root, TEST_CLASS_WITH_PYTHON, morpheus

#include "morpheus/controllers/monitor_controller.hpp"  // for MonitorController
#include "morpheus/messages/control.hpp"                // for ControlMessage

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(MonitorController);

cudf::io::table_with_metadata create_cudf_table_with_metadata(int rows, int cols)
{
    std::vector<std::unique_ptr<cudf::column>> columns;

    for (int i = 0; i < cols; ++i)
    {
        auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, rows);
        auto col_view = col->mutable_view();

        std::vector<int32_t> data(rows);
        std::iota(data.begin(), data.end(), 0);
        cudaMemcpy(col_view.data<int32_t>(), data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

        columns.push_back(std::move(col));
    }

    auto table = std::make_unique<cudf::table>(std::move(columns));

    auto index_info   = cudf::io::column_name_info{""};
    auto column_names = std::vector<cudf::io::column_name_info>(cols, index_info);
    auto metadata     = cudf::io::table_metadata{std::move(column_names), {}, {}};

    return cudf::io::table_with_metadata{std::move(table), metadata};
}

TEST_F(TestMonitorController, TestAutoCountFn)
{
    auto message_meta_mc            = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta");
    auto message_meta_auto_count_fn = message_meta_mc.auto_count_fn();
    auto meta                       = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    EXPECT_EQ((*message_meta_auto_count_fn)(meta), 10);

    auto control_message_mc            = MonitorController<std::shared_ptr<ControlMessage>>("test_control_message");
    auto control_message_auto_count_fn = control_message_mc.auto_count_fn();
    auto control_message               = std::make_shared<ControlMessage>();
    auto cm_meta = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(20, 3)));
    control_message->payload(cm_meta);
    EXPECT_EQ((*control_message_auto_count_fn)(control_message), 20);

    auto message_meta_vector_mc =
        MonitorController<std::vector<std::shared_ptr<MessageMeta>>>("test_message_meta_vector");
    auto message_meta_vector_auto_count_fn = message_meta_vector_mc.auto_count_fn();
    std::vector<std::shared_ptr<MessageMeta>> meta_vector;
    for (int i = 0; i < 5; ++i)
    {
        meta_vector.emplace_back(MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(5, 2))));
    }
    EXPECT_EQ((*message_meta_vector_auto_count_fn)(meta_vector), 25);

    auto control_message_vector_mc =
        MonitorController<std::vector<std::shared_ptr<ControlMessage>>>("test_control_message_vector");
    auto control_message_vector_auto_count_fn = control_message_vector_mc.auto_count_fn();
    std::vector<std::shared_ptr<ControlMessage>> control_message_vector;
    for (int i = 0; i < 5; ++i)
    {
        auto cm = std::make_shared<ControlMessage>();
        cm->payload(MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(6, 2))));
        control_message_vector.emplace_back(cm);
    }
    EXPECT_EQ((*control_message_vector_auto_count_fn)(control_message_vector), 30);

    // Test invalid message type
    EXPECT_THROW(MonitorController<int>("invalid message type"), std::runtime_error);
}

TEST_F(TestMonitorController, TestProgressBar)
{
    auto message_meta_mc   = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta");
    auto meta              = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    auto message_meta_mc_2 = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta_2");
    auto meta_2            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    auto message_meta_mc_3 = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta_3");
    auto meta_3            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));

    auto control_message_mc = MonitorController<std::shared_ptr<ControlMessage>>("test_control_message");
    auto control_message    = std::make_shared<ControlMessage>();
    auto cm_meta            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(20, 3)));
    control_message->payload(cm_meta);

    for (int i = 0; i < 10; i++)
    {
        // std::cout << "log message" << std::endl;
        message_meta_mc.progress_sink(meta);
        // std::cout << "log message 2" << std::endl;
        message_meta_mc_2.progress_sink(meta_2);
        // std::cout << "log message 3" << std::endl;
        message_meta_mc_3.progress_sink(meta_3);
        control_message_mc.progress_sink(control_message);
        std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::milliseconds(100));
    }
    // using namespace indicators;
    // auto* stdout_buf = std::cout.rdbuf();  // get stdout streambuf

    // // create a filtering_ostreambuf with our filter and the stdout streambuf as a sink
    // boost::iostreams::filtering_ostreambuf filtering_buf{};
    // filtering_buf.push(LineInsertingFilter());
    // filtering_buf.push(*stdout_buf);

    // std::cout.rdbuf(&filtering_buf);  // configure std::cout to use our streambuf

    // std::ostream os(stdout_buf);  // create local ostream acting as std::cout normally would

    // auto bar1 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::red},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"5c90d4a2d1a8: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // auto bar2 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::yellow},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"22337bfd13a9: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // auto bar3 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::green},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"10f26c680a34: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // auto bar4 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::white},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"6364e0d7a283: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // auto bar5 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::blue},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"ff1356ba118b: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // auto bar6 = std::make_unique<ProgressBar>(
    //     option::BarWidth{50},
    //     option::ForegroundColor{Color::cyan},
    //     option::ShowElapsedTime{true},
    //     option::ShowRemainingTime{true},
    //     option::PrefixText{"5a17453338b4: Downloading "},
    //     indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
    //     option::Stream{os});

    // std::cout << termcolor::bold << termcolor::white << "Pulling image foo:bar/baz\n";

    // // Construct with 3 progress bars. We'll add 3 more at a later point
    // DynamicProgress<ProgressBar> bars(*bar1, *bar2, *bar3);

    // // Do not hide bars when completed
    // bars.set_option(option::HideBarWhenComplete{false});

    // std::thread fourth_job, fifth_job, sixth_job;

    // auto job4 = [&bars](size_t i) {
    //     while (true)
    //     {
    //         bars[i].tick();
    //         if (bars[i].is_completed())
    //         {
    //             bars[i].set_option(option::PrefixText{"6364e0d7a283: Pull complete "});
    //             bars[i].mark_as_completed();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //     }
    // };

    // auto job5 = [&bars](size_t i) {
    //     while (true)
    //     {
    //         bars[i].tick();
    //         if (bars[i].is_completed())
    //         {
    //             bars[i].set_option(option::PrefixText{"ff1356ba118b: Pull complete "});
    //             bars[i].mark_as_completed();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //     }
    // };

    // auto job6 = [&bars](size_t i) {
    //     while (true)
    //     {
    //         bars[i].tick();
    //         if (bars[i].is_completed())
    //         {
    //             bars[i].set_option(option::PrefixText{"5a17453338b4: Pull complete "});
    //             bars[i].mark_as_completed();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(40));
    //     }
    // };

    // auto job1 = [&bars, &bar6, &sixth_job, &job6]() {
    //     while (true)
    //     {
    //         bars[0].tick();
    //         if (bars[0].is_completed())
    //         {
    //             bars[0].set_option(option::PrefixText{"5c90d4a2d1a8: Pull complete "});
    //             // bar1 is completed, adding bar6
    //             auto i    = bars.push_back(*bar6);
    //             sixth_job = std::thread(job6, i);
    //             sixth_job.join();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(140));
    //     }
    // };

    // auto job2 = [&bars, &bar5, &fifth_job, &job5]() {
    //     while (true)
    //     {
    //         bars[1].tick();
    //         if (bars[1].is_completed())
    //         {
    //             bars[1].set_option(option::PrefixText{"22337bfd13a9: Pull complete "});
    //             // bar2 is completed, adding bar5
    //             auto i    = bars.push_back(*bar5);
    //             fifth_job = std::thread(job5, i);
    //             fifth_job.join();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(25));
    //     }
    // };

    // auto job3 = [&bars, &bar4, &fourth_job, &job4]() {
    //     while (true)
    //     {
    //         bars[2].tick();
    //         if (bars[2].is_completed())
    //         {
    //             bars[2].set_option(option::PrefixText{"10f26c680a34: Pull complete "});
    //             // bar3 is completed, adding bar4
    //             auto i     = bars.push_back(*bar4);
    //             fourth_job = std::thread(job4, i);
    //             fourth_job.join();
    //             break;
    //         }
    //         std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //     }
    // };

    // std::thread first_job(job1);
    // std::thread second_job(job2);
    // std::thread third_job(job3);

    // third_job.join();
    // second_job.join();
    // first_job.join();

    // std::cout << termcolor::bold << termcolor::green << "✔ Downloaded image foo/bar:baz" << std::endl;
    // std::cout << termcolor::reset;
}
