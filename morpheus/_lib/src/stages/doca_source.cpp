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

#include "morpheus/stages/doca_source.hpp"
#include "morpheus/stages/doca_source_kernels.hpp"
#include "morpheus/doca/common.h"

#include <cudf/column/column.hpp>  // for column
#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table.hpp>                  // for table
#include <cudf/types.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <glog/logging.h>
#include <cuda/std/chrono>

#include <rmm/device_uvector.hpp>

#include <algorithm>  // for find
#include <cstddef>    // for size_t
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <utility>
#include <iostream>

std::optional<uint32_t> ip_to_int(std::string const& ip_address)
{
  if (ip_address.empty())
  {
    return 0;
  }

  uint8_t a, b, c, d;
  uint32_t ret;

  ret = sscanf(ip_address.c_str(), "%hhu.%hhu.%hhu.%hhu", &a, &b, &c, &d);

  printf("%u: %u %u %u %u\n", ret, a, b, c, d);

  if (ret == 4)
  {
    return BE_IPV4_ADDR(a, b, c, d);
  }

  return std::nullopt;
}

namespace morpheus {
// Component public implementations
// ************ DocaSourceStage ************* //
DocaSourceStage::DocaSourceStage(
  std::string const& nic_pci_address,
  std::string const& gpu_pci_address,
  std::string const& source_ip_filter
) :
  PythonSource(build())
{
  auto source_ip = ip_to_int(source_ip_filter);

  if (source_ip == std::nullopt) {
    throw std::runtime_error("source ip filter invalid");
  }

  _context   = std::make_shared<morpheus::doca::doca_context>(
    nic_pci_address, // "17:00.1"
    gpu_pci_address  // "ca:00.0"
  );

  _rxq       = std::make_shared<morpheus::doca::doca_rx_queue>(_context);
  _rxpipe    = std::make_shared<morpheus::doca::doca_rx_pipe>(_context, _rxq, source_ip.value());
  _semaphore = std::make_shared<morpheus::doca::doca_semaphore>(_context, 1024);
}

DocaSourceStage::subscriber_fn_t DocaSourceStage::build()
{
  return [this](rxcpp::subscriber<source_type_t> output) {

    cudaStream_t processing_stream;
    cudaStreamCreateWithFlags(&processing_stream, cudaStreamNonBlocking);

    auto semaphore_idx_d    = rmm::device_scalar<int32_t>(0, processing_stream);
    auto packet_count_d     = rmm::device_scalar<int32_t>(0, processing_stream);
    auto packet_data_size_d = rmm::device_scalar<int32_t>(0, processing_stream);

    while (output.is_subscribed())
    {
      morpheus::doca::packet_receive_kernel(
        _rxq->rxq_info_gpu(),
        _semaphore->in_gpu(),
        _semaphore->size(),
        semaphore_idx_d.data(),
        packet_count_d.data(),
        packet_data_size_d.data(),
        processing_stream
      );

      auto packet_count = packet_count_d.value(processing_stream);

      if (packet_count == 0)
      {
        continue;
      }

      auto packet_data_size = packet_data_size_d.value(processing_stream);

      auto timestamp_out_d     = rmm::device_uvector<uint32_t>(packet_count, processing_stream);
      auto src_mac_out_d       = rmm::device_uvector<int64_t>(packet_count, processing_stream);
      auto dst_mac_out_d       = rmm::device_uvector<int64_t>(packet_count, processing_stream);
      auto src_ip_out_d        = rmm::device_uvector<int64_t>(packet_count, processing_stream);
      auto dst_ip_out_d        = rmm::device_uvector<int64_t>(packet_count, processing_stream);
      auto src_port_out_d      = rmm::device_uvector<uint16_t>(packet_count, processing_stream);
      auto dst_port_out_d      = rmm::device_uvector<uint16_t>(packet_count, processing_stream);
      auto data_offsets_out_d  = rmm::device_uvector<int32_t>(packet_count + 1, processing_stream);
      auto data_size_out_d     = rmm::device_uvector<int32_t>(packet_count, processing_stream);
      auto tcp_flags_out_d     = rmm::device_uvector<int32_t>(packet_count, processing_stream);
      auto ether_type_out_d    = rmm::device_uvector<int32_t>(packet_count, processing_stream);
      auto next_proto_id_out_d = rmm::device_uvector<int32_t>(packet_count, processing_stream);
      auto data_out_d          = rmm::device_uvector<char>(packet_data_size, processing_stream);

      data_offsets_out_d.set_element_async(packet_count, packet_data_size, processing_stream);

      morpheus::doca::packet_gather_kernel(
        _rxq->rxq_info_gpu(),
        _semaphore->in_gpu(),
        _semaphore->size(),
        semaphore_idx_d.data(),
        timestamp_out_d.data(),
        src_mac_out_d.data(),
        dst_mac_out_d.data(),
        src_ip_out_d.data(),
        dst_ip_out_d.data(),
        src_port_out_d.data(),
        dst_port_out_d.data(),
        data_offsets_out_d.data(),
        data_size_out_d.data(),
        tcp_flags_out_d.data(),
        ether_type_out_d.data(),
        next_proto_id_out_d.data(),
        data_out_d.data(),
        processing_stream
      );

      auto sem_idx_old = semaphore_idx_d.value(processing_stream);
      auto sem_idx_new = (sem_idx_old + 1) % _semaphore->size();
      semaphore_idx_d.set_value_async(sem_idx_new, processing_stream);

      // int32_t last_offset = data_offsets_out_d.back_element(processing_stream);

      // std::cout << "sem_idx:     "      << sem_idx_old      << std::endl
      //           << "packet_count:     " << packet_count     << std::endl
      //           << "packet_data_size: " << packet_data_size << std::endl
      //           << "last_offset:      " << last_offset      << std::endl
      //           << std::flush;

      // data columns
      auto data_offsets_out_d_size = data_offsets_out_d.size();
      auto data_offsets_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        data_offsets_out_d_size,
        data_offsets_out_d.release());

      auto data_out_d_size = data_out_d.size();
      auto data_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int8_t>()},
        data_out_d_size,
        data_out_d.release());

      auto data_col = cudf::make_strings_column(
        packet_count,
        std::move(data_offsets_out_d_col),
        std::move(data_out_d_col),
        0,
        {});

      // timestamp column
      auto timestamp_out_d_size = timestamp_out_d.size();
      auto timestamp_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<uint32_t>()},
        timestamp_out_d_size,
        timestamp_out_d.release());

      // src_mac address column
      auto src_mac_out_d_size = src_mac_out_d.size();
      auto src_mac_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int64_t>()},
        src_mac_out_d_size,
        src_mac_out_d.release());
      auto src_mac_out_str_col = morpheus::doca::integers_to_mac(src_mac_out_d_col->view(), processing_stream);

      // dst_mac address column
      auto dst_mac_out_d_size = dst_mac_out_d.size();
      auto dst_mac_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int64_t>()},
        dst_mac_out_d_size,
        dst_mac_out_d.release());
      auto dst_mac_out_str_col = morpheus::doca::integers_to_mac(dst_mac_out_d_col->view(), processing_stream);

      cudaStreamSynchronize(processing_stream);

      // src ip address column
      auto src_ip_out_d_size = src_ip_out_d.size();
      auto src_ip_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int64_t>()},
        src_ip_out_d_size,
        src_ip_out_d.release());
      auto src_ip_out_str_col = cudf::strings::integers_to_ipv4(src_ip_out_d_col->view());

      // dst ip address column
      auto dst_ip_out_d_size = dst_ip_out_d.size();
      auto dst_ip_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int64_t>()},
        dst_ip_out_d_size,
        dst_ip_out_d.release());
      auto dst_ip_out_str_col = cudf::strings::integers_to_ipv4(dst_ip_out_d_col->view());

      // src port column
      auto src_port_out_d_size = src_port_out_d.size();
      auto src_port_out_d_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<uint16_t>()},
        src_port_out_d_size,
        src_port_out_d.release());

      // dst port column
      auto dst_port_out_d_size = dst_port_out_d.size();
      auto dst_port_out_d_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<uint16_t>()},
        dst_port_out_d_size,
        dst_port_out_d.release());

      // packet size column
      auto data_size_out_d_size = data_size_out_d.size();
      auto data_size_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        data_size_out_d_size,
        data_size_out_d.release());

      // tcp flags column
      auto tcp_flags_out_d_size = tcp_flags_out_d.size();
      auto tcp_flags_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        tcp_flags_out_d_size,
        tcp_flags_out_d.release());

      // frame type column
      auto ether_type_out_d_size = ether_type_out_d.size();
      auto ether_type_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        ether_type_out_d_size,
        ether_type_out_d.release());

      // protocol id column
      auto next_proto_id_out_d_size = next_proto_id_out_d.size();
      auto next_proto_id_out_d_col  = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        next_proto_id_out_d_size,
        next_proto_id_out_d.release());

      // create dataframe

      auto my_columns = std::vector<std::unique_ptr<cudf::column>>();
      auto metadata = cudf::io::table_metadata();

      metadata.column_names.push_back("timestamp");
      my_columns.push_back(std::move(timestamp_out_d_col));

      metadata.column_names.push_back("src_mac");
      my_columns.push_back(std::move(src_mac_out_str_col));

      metadata.column_names.push_back("dst_mac");
      my_columns.push_back(std::move(dst_mac_out_str_col));

      metadata.column_names.push_back("src_ip");
      my_columns.push_back(std::move(src_ip_out_str_col));

      metadata.column_names.push_back("dst_ip");
      my_columns.push_back(std::move(dst_ip_out_str_col));

      metadata.column_names.push_back("src_port");
      my_columns.push_back(std::move(src_port_out_d_col));

      metadata.column_names.push_back("dst_port");
      my_columns.push_back(std::move(dst_port_out_d_col));

      metadata.column_names.push_back("packet_size");
      my_columns.push_back(std::move(data_size_out_d_col));

      metadata.column_names.push_back("tcp_flags");
      my_columns.push_back(std::move(tcp_flags_out_d_col));

      metadata.column_names.push_back("ether_type");
      my_columns.push_back(std::move(ether_type_out_d_col));

      metadata.column_names.push_back("next_proto_id");
      my_columns.push_back(std::move(next_proto_id_out_d_col));

      metadata.column_names.push_back("data");
      my_columns.push_back(std::move(data_col));

      auto my_table_w_metadata = cudf::io::table_with_metadata{
        std::make_unique<cudf::table>(std::move(my_columns)),
        std::move(metadata)
      };

      auto meta = MessageMeta::create_from_cpp(std::move(my_table_w_metadata), 0);

      output.on_next(std::move(meta));
    }

    output.on_completed();
  };
}

// ************ DocaSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<DocaSourceStage>> DocaSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    std::string const& name,
    std::string const& nic_pci_address,
    std::string const& gpu_pci_address,
    std::string const& source_ip_filter)
{
    return builder.construct_object<DocaSourceStage>(
      name,
      nic_pci_address,
      gpu_pci_address,
      source_ip_filter
    );
}

}  // namespace morpheus
