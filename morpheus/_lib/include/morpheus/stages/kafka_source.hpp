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

#include "morpheus/messages/meta.hpp"

#include <cudf/io/types.hpp>
#include <librdkafka/rdkafkacpp.h>
#include <pysrf/node.hpp>
#include <srf/core/fiber_meta_data.hpp>
#include <srf/core/task_queue.hpp>
#include <srf/segment/builder.hpp>

#include <memory>
#include <string>
#include <vector>

namespace morpheus {

/****** Component public implementations *******************/
/****** KafkaSourceStage************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)

class KafkaSourceStage : public srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    KafkaSourceStage(size_t max_batch_size,
                     std::string topic,
                     int32_t batch_timeout_ms,
                     std::map<std::string, std::string> config,
                     bool disable_commit        = false,
                     bool disable_pre_filtering = false,
                     size_t stop_after          = 0);

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
    subscriber_fn_t build();

    /**
     * TODO(Documentation)
     */
    std::unique_ptr<RdKafka::Conf> build_kafka_conf(const std::map<std::string, std::string> &config_in);

    /**
     * TODO(Documentation)
     */
    std::unique_ptr<RdKafka::KafkaConsumer> create_consumer(RdKafka::RebalanceCb &rebalancer);

    /**
     * TODO(Documentation)
     */
    cudf::io::table_with_metadata load_table(const std::string &buffer);

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<morpheus::MessageMeta> process_batch(
        std::vector<std::unique_ptr<RdKafka::Message>> &&message_batch);

    size_t m_max_batch_size{128};
    uint32_t m_batch_timeout_ms{100};

    std::string m_topic{"test_pcap"};
    std::map<std::string, std::string> m_config;

    bool m_disable_commit{false};
    bool m_disable_pre_filtering{false};
    bool m_requires_commit{false};  // Whether or not manual committing is required
    size_t m_stop_after{0};

    void *m_rebalancer;
};

/****** KafkaSourceStageInferenceProxy**********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct KafkaSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a KafkaSourceStage, and return the result.
     */
    static std::shared_ptr<srf::segment::Object<KafkaSourceStage>> init(srf::segment::Builder &builder,
                                                                        const std::string &name,
                                                                        size_t max_batch_size,
                                                                        std::string topic,
                                                                        int32_t batch_timeout_ms,
                                                                        std::map<std::string, std::string> config,
                                                                        bool disable_commits,
                                                                        bool disable_pre_filtering,
                                                                        size_t stop_after = 0);
};
#pragma GCC visibility pop
}  // namespace morpheus
