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

#include <cudf/column/column.hpp>  // for column
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table.hpp>                  // for table
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/cast.h>  // for object_api::operator()
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_
#include <srf/segment/builder.hpp>
#include <cuda/atomic>
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
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ DocaSourceStage ************* //
DocaSourceStage::DocaSourceStage() :
  PythonSource(build())
{
  _context   = std::make_shared<morpheus::doca::doca_context>("b5:00.0", "b6:00.0");
  _rxq       = std::make_shared<morpheus::doca::doca_rx_queue>(_context);
  _rxpipe    = std::make_shared<morpheus::doca::doca_rx_pipe>(_context, _rxq);
  _semaphore = std::make_shared<morpheus::doca::doca_semaphore>(_context, 32);
}

DocaSourceStage::subscriber_fn_t DocaSourceStage::build()
{
  return [this](rxcpp::subscriber<source_type_t> output) {

    cudaStream_t receiving_stream;
    cudaStream_t processing_stream;
    cudaStreamCreateWithFlags(&receiving_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&processing_stream, cudaStreamNonBlocking);

    auto exit_flag = rmm::device_scalar<cuda::atomic<bool>>{false, receiving_stream};

    doca_packet_receive_persistent_kernel(
      _rxq->rxq_info_gpu(),
      _semaphore->in_gpu(),
      _semaphore->size(),
      exit_flag.data(),
      receiving_stream
    );

    auto packet_count_d  = rmm::device_scalar<uint32_t>(0, receiving_stream);
    auto packets_size_d  = rmm::device_scalar<uint32_t>(0, receiving_stream);
    auto sem_idx_begin_d = rmm::device_scalar<uint32_t>(0, receiving_stream);
    auto sem_idx_end_d   = rmm::device_scalar<uint32_t>(0, receiving_stream);

    while (output.is_subscribed())
    {
      cudaStreamSynchronize(processing_stream);

      doca_packet_count_kernel(
        _rxq->rxq_info_gpu(),
        _semaphore->in_gpu(),
        _semaphore->size(),
        sem_idx_begin_d.data(),
        sem_idx_end_d.data(),
        cuda::std::chrono::duration<int64_t>(1),
        2048,
        packet_count_d.data(),
        packets_size_d.data(),
        exit_flag.data(),
        processing_stream
      );

      cudaStreamSynchronize(processing_stream);

      auto packet_count = packet_count_d.value(processing_stream);

      printf("host: num packets received: %d\n", packet_count);

      // allocate stuff

      doca_packet_gather_kernel(
        _rxq->rxq_info_gpu(),
        _semaphore->in_gpu(),
        _semaphore->size(),
        sem_idx_begin_d.data(),
        sem_idx_end_d.data(),
        packet_count_d.data(),
        packets_size_d.data(),
        exit_flag.data(),
        processing_stream
      );
    }

    auto flag = cuda::atomic<bool>(true);
    exit_flag.set_value_async(flag, processing_stream);

    cudaStreamSynchronize(processing_stream);
    cudaStreamSynchronize(receiving_stream);
    cudaStreamDestroy(receiving_stream);

    auto cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
      std::cout << "cuda error: " << cudaGetErrorString(cuda_error) << std::endl;
      return;
    }

    {
      auto stream = cudf::default_stream_value;
      auto values = rmm::device_uvector<int32_t>(1000000, stream);

      auto values_size = values.size();
      auto my_column   = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<int32_t>()},
        values_size,
        values.release());

      auto my_columns          = std::vector<std::unique_ptr<cudf::column>>();

      my_columns.push_back(std::move(my_column));

      // auto my_table            = std::make_unique<cudf::table>(std::move(my_columns));
      auto metadata            = cudf::io::table_metadata();

      metadata.column_names.push_back("index");

      auto my_table_w_metadata = cudf::io::table_with_metadata{
        std::make_unique<cudf::table>(std::move(my_columns)),
        std::move(metadata)
      };
      auto meta                = MessageMeta::create_from_cpp(std::move(my_table_w_metadata), 1);

      output.on_next(meta);
    }

    output.on_completed();
  };
}

// ************ DocaSourceStageInterfaceProxy ************ //
std::shared_ptr<srf::segment::Object<DocaSourceStage>> DocaSourceStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name)
{
    auto stage = builder.construct_object<DocaSourceStage>(name);

    return stage;
}
}  // namespace morpheus
