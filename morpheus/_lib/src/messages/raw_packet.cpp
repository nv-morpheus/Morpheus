/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/raw_packet.hpp"

#include <pybind11/pytypes.h>

#include <memory>
// We're already including pybind11.h and don't need to include cast.
// For some reason IWYU also thinks we need array for the `isinsance` call.
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <array>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

/****** Component public implementations *******************/
/****** RawPacketMessage ****************************************/

uint32_t RawPacketMessage::count() const
{
    return num;
}

uint32_t RawPacketMessage::get_max_size() const
{
    return max_size;
}

uintptr_t RawPacketMessage::get_pkt_addr_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > num || gpu_mem == true)
        return 0;
    return ptr_addr[pkt_idx];
}

uintptr_t RawPacketMessage::get_pkt_hdr_size_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > num || gpu_mem == true)
        return 0;
    return ptr_hdr_size[pkt_idx];
}

uintptr_t RawPacketMessage::get_pkt_pld_size_idx(uint32_t pkt_idx) const
{
    if (pkt_idx > num || gpu_mem == true)
        return 0;
    return ptr_pld_size[pkt_idx];
}

uintptr_t* RawPacketMessage::get_pkt_addr_list() const
{
    return ptr_addr;
}

uint32_t* RawPacketMessage::get_pkt_hdr_size_list() const
{
    return ptr_hdr_size;
}

uint32_t* RawPacketMessage::get_pkt_pld_size_list() const
{
    return ptr_pld_size;
}

uint32_t RawPacketMessage::get_queue_idx() const
{
    return queue_idx;
}

bool RawPacketMessage::is_gpu_mem() const
{
    return gpu_mem;
}

std::shared_ptr<RawPacketMessage> RawPacketMessage::create_from_cpp(uint32_t num,
                                                                    uint32_t max_size,
                                                                    uintptr_t* ptr_addr,
                                                                    uint32_t* ptr_hdr_size,
                                                                    uint32_t* ptr_pld_size,
                                                                    bool gpu_mem,
                                                                    uint16_t queue_idx)
{
    return std::shared_ptr<RawPacketMessage>(
        new RawPacketMessage(num, max_size, ptr_addr, ptr_hdr_size, ptr_pld_size, gpu_mem, queue_idx));
}

// std::shared_ptr<RawPacketMessage> RawPacketMessage::create_from_python(py::object&& data_table)
// {
//     auto data = std::make_unique<PyDataTable>(std::move(data_table));

//     return std::shared_ptr<RawPacketMessage>(new RawPacketMessage(std::move(data)));
// }

RawPacketMessage::RawPacketMessage(uint32_t num_,
                                   uint32_t max_size_,
                                   uintptr_t* ptr_addr_,
                                   uint32_t* ptr_hdr_size_,
                                   uint32_t* ptr_pld_size_,
                                   bool gpu_mem_,
                                   int queue_idx_) :
  num(num_),
  max_size(max_size_),
  ptr_addr(ptr_addr_),
  ptr_hdr_size(ptr_hdr_size_),
  ptr_pld_size(ptr_pld_size_),
  gpu_mem(gpu_mem_),
  queue_idx(queue_idx_)
{}

// py::object RawPacketMessage::cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count)
// {
//     py::gil_scoped_acquire gil;

//     // Now convert to a python TableInfo object
//     auto converted_table = CudfHelper::table_from_table_with_metadata(std::move(table), index_col_count);

//     // VLOG(10) << "Table. Num Col: " << converted_table.attr("_num_columns").str().cast<std::string>()
//     //          << ", Num Ind: " << converted_table.attr("_num_columns").cast<std::string>()
//     //          << ", Rows: " << converted_table.attr("_num_rows").cast<std::string>();
//     // py::print("Table Created. Num Rows: {}, Num Cols: {}, Num Ind: {}",
//     //           converted_table.attr("_num_rows"),
//     //           converted_table.attr("_num_columns"),
//     //           converted_table.attr("_num_indices"));

//     return converted_table;
// }

}  // namespace morpheus
