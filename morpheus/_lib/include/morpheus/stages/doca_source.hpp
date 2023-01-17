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

#include <morpheus/doca/doca_context.hpp>
#include <morpheus/messages/meta.hpp>

#include <cudf/io/types.hpp>  // for table_with_metadata
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity
#include <mrc/channel/status.hpp>          // for Status
#include <mrc/node/source_properties.hpp>  // for SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object

#include <memory>
#include <string>
#include <vector>  // for vector

namespace morpheus {
/****** Component public implementations *******************/
/****** DocaSourceStage*************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class DocaSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    DocaSourceStage(
      std::string const& nic_pci_address,
      std::string const& gpu_pci_address,
      std::string const& source_ip_filter
    );

  private:
    subscriber_fn_t build();

    std::shared_ptr<morpheus::doca::doca_context> _context;
    std::shared_ptr<morpheus::doca::doca_rx_queue> _rxq;
    std::shared_ptr<morpheus::doca::doca_rx_pipe> _rxpipe;
    std::shared_ptr<morpheus::doca::doca_semaphore> _semaphore;
};

/****** DocaSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DocaSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DocaSourceStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<DocaSourceStage>> init(
      mrc::segment::Builder& builder,
      std::string const& name,
      std::string const& nic_pci_address,
      std::string const& gpu_pci_address,
      std::string const& source_ip_filter);
};
#pragma GCC visibility pop
}  // namespace morpheus
