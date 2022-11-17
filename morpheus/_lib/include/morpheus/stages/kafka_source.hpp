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
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity
#include <srf/channel/status.hpp>          // for Status
#include <srf/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <srf/segment/builder.hpp>
#include <srf/segment/object.hpp>  // for Object

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, uint32_t
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {

/****** Component public implementations *******************/
/****** KafkaSourceStage************************************/

/**
 * @addtogroup stages
 * @{
 * @file
*/

#pragma GCC visibility push(default)
/**
 * This class loads messages from the Kafka cluster by serving as a Kafka consumer.
 */
class KafkaSourceStage : public srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = srf::pysrf::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    /**
     * @brief Construct a new Kafka Source Stage object
     * 
     * @param max_batch_size : The maximum batch size for the messages batch. 
     * @param topic : Input kafka topic.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server. 
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages. 
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table). 
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     */
    KafkaSourceStage(size_t max_batch_size,
                     std::string topic,
                     int32_t batch_timeout_ms,
                     std::map<std::string, std::string> config,
                     bool disable_commit        = false,
                     bool disable_pre_filtering = false,
                     size_t stop_after          = 0,
                     bool async_commits         = true);

    ~KafkaSourceStage() override = default;

    /**
     * @return maximum batch size for KafkaSource.
     */
    std::size_t max_batch_size();

    /**
     * @return batch timeout in ms.
     */
    int32_t batch_timeout_ms();

  private:
    /**
     * TODO(Documentation)
     */
    subscriber_fn_t build();

    /**
     * @brief Create kafka consumer configuration and returns unique pointer to the result.
     * 
     * @param config_in : Configuration map contains Kafka consumer properties.
     * @return std::unique_ptr<RdKafka::Conf>
     */
    std::unique_ptr<RdKafka::Conf> build_kafka_conf(const std::map<std::string, std::string> &config_in);

    /**
     * @brief Creates Kafka consumer instance.
     * 
     * @param rebalancer : Group rebalance callback for use with RdKafka::KafkaConsumer.
     * @return std::unique_ptr<RdKafka::KafkaConsumer>
     */
    std::unique_ptr<RdKafka::KafkaConsumer> create_consumer(RdKafka::RebalanceCb &rebalancer);

    /**
     * @brief Load messages from a buffer/file to a cuDF table.
     * 
     * @param buffer : Reference of a messages buffer
     * @return cudf::io::table_with_metadata
     */
    cudf::io::table_with_metadata load_table(const std::string &buffer);

    /**
     * @brief This function combines JSON messages from Kafka, parses them, then loads them onto a MessageMeta.
     * and returns the shared pointer as a result.
     * 
     * @param message_batch : Reference of a message batch that needs to be processed.
     * @return std::shared_ptr<morpheus::MessageMeta>
     */
    std::shared_ptr<morpheus::MessageMeta> process_batch(
        std::vector<std::unique_ptr<RdKafka::Message>> &&message_batch);

    size_t m_max_batch_size{128};
    uint32_t m_batch_timeout_ms{100};

    std::string m_topic;
    std::map<std::string, std::string> m_config;

    bool m_disable_commit{false};
    bool m_disable_pre_filtering{false};
    bool m_requires_commit{false};  // Whether or not manual committing is required
    bool m_async_commits{true};
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
     * @brief Create and initialize a KafkaSourceStage, and return the result
     * 
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param max_batch_size : The maximum batch size for the messages batch. 
     * @param topic : Input kafka topic.
     * @param batch_timeout_ms : Frequency of the poll in ms.
     * @param config : Kafka consumer configuration.
     * @param disable_commit : Enabling this option will skip committing messages as they are pulled off the server. 
     * This is only useful for debugging, allowing the user to process the same messages multiple times
     * @param disable_pre_filtering : Enabling this option will skip pre-filtering of json messages. 
     * This is only useful when inputs are known to be valid json.
     * @param stop_after : Stops ingesting after emitting `stop_after` records (rows in the table).
     * Useful for testing. Disabled if `0`
     * @param async_commits : Asynchronously acknowledge consuming Kafka messages
     */
    static std::shared_ptr<srf::segment::Object<KafkaSourceStage>> init(srf::segment::Builder &builder,
                                                                        const std::string &name,
                                                                        size_t max_batch_size,
                                                                        std::string topic,
                                                                        int32_t batch_timeout_ms,
                                                                        std::map<std::string, std::string> config,
                                                                        bool disable_commits,
                                                                        bool disable_pre_filtering,
                                                                        size_t stop_after  = 0,
                                                                        bool async_commits = true);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
