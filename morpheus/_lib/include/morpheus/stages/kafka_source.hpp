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

#include <morpheus/messages/meta.hpp>

#include <neo/core/fiber_meta_data.hpp>
#include <neo/core/task_queue.hpp>
#include <pyneo/node.hpp>

#include <cudf/io/types.hpp>
#include <librdkafka/rdkafkacpp.h>

#include <string>
#include <memory>
#include <vector>


namespace morpheus {

    /****** Component public implementations *******************/
    /****** KafkaSourceStage************************************/
    /**
     * TODO(Documentation)
     */
#pragma GCC visibility push(default)
    class KafkaSourceStage : public neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>> {
    public:
        using base_t = neo::pyneo::PythonSource<std::shared_ptr<MessageMeta>>;
        using base_t::source_type_t;

        KafkaSourceStage(const neo::Segment &parent,
                         const std::string &name,
                         size_t max_batch_size,
                         std::string topic,
                         int32_t batch_timeout_ms,
                         std::map<std::string, std::string> config,
                         bool disable_commit = false,
                         bool disable_pre_filtering = false);

        ~KafkaSourceStage() override = default;


        /**
         * @return maximum batch size for KafkaSource
         */
        std::size_t max_batch_size();

        /**
         * @return batch timeout in ms
         */
        int32_t batch_timeout_ms();

    private:
        /**
         * TODO(Documentation)
         */
        void start() override;

        /**
         * TODO(Documentation)
         */
        std::unique_ptr<RdKafka::Conf> build_kafka_conf(const std::map<std::string, std::string> &config_in);

        /**
         * TODO(Documentation)
         */
        neo::SharedFuture<bool> launch_tasks(std::vector<std::function<bool()>> &&tasks);

        /**
         * TODO(Documentation)
         */
        std::unique_ptr<RdKafka::KafkaConsumer> create_consumer();

        /**
         * TODO(Documentation)
         */
        cudf::io::table_with_metadata load_table(const std::string &buffer);

        /**
         * TODO(Documentation)
         */
        std::shared_ptr<morpheus::MessageMeta>
        process_batch(std::vector<std::unique_ptr<RdKafka::Message>> &&message_batch);

        size_t m_max_batch_size{128};
        uint32_t m_batch_timeout_ms{100};

        std::string m_topic{"test_pcap"};
        std::map<std::string, std::string> m_config;

        bool m_disable_commit{false};
        bool m_disable_pre_filtering{false};
        bool m_requires_commit{false};  // Whether or not manual committing is required
        std::vector<std::shared_ptr<neo::TaskQueue<neo::FiberMetaData>>> m_task_queues;

        void *m_rebalancer;
    };

    /****** KafkaSourceStageInferenceProxy**********************/
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct KafkaSourceStageInterfaceProxy {
        /**
         * @brief Create and initialize a KafkaSourceStage, and return the result.
         */
        static std::shared_ptr<KafkaSourceStage> init(
                neo::Segment &parent,
                const std::string &name,
                size_t max_batch_size,
                std::string topic,
                int32_t batch_timeout_ms,
                std::map<std::string, std::string> config,
                bool disable_commits,
                bool disable_pre_filtering);
    };
#pragma GCC visibility pop
}