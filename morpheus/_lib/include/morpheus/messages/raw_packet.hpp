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

#pragma once

#include <pybind11/pytypes.h>  // for object, dict, list, none

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morpheus {

#pragma GCC visibility push(default)
/****** Component public implementations ******************/
/****** RawPacketMessage ****************************************/

/**
 * @brief Container for class holding a list of raw packets (number of packets, max size and pointers)
 *
 */
class RawPacketMessage
{
  public:
    /**
     * @brief Return number of packets in the message
     *
     * @return uint32_t
     */
    uint32_t count() const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uint32_t
     */
    uint32_t get_max_size() const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t
     */
    uintptr_t get_pkt_addr_idx(uint32_t pkt_idx) const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t
     */
    uintptr_t get_pkt_hdr_size_idx(uint32_t pkt_idx) const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t
     */
    uintptr_t get_pkt_pld_size_idx(uint32_t pkt_idx) const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t *
     */
    uintptr_t * get_pkt_addr_list() const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t *
     */
    uint32_t * get_pkt_hdr_size_list() const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uintptr_t *
     */
    uint32_t * get_pkt_pld_size_list() const;

    /**
     * @brief Return max packet size in the message
     *
     * @return uint32_t
     */
    uint32_t get_queue_idx() const;

    /**
     * @brief Return if packet list is store in GPU (true) or CPU pinned memory (false)
     *
     * @return bool
     */
    bool is_gpu_mem() const;

    // /**
    //  * @brief Create RawPacketMessage cpp object from a python object
    //  *
    //  * @param data_table
    //  * @return std::shared_ptr<RawPacketMessage>
    //  */
    // static std::shared_ptr<RawPacketMessage> create_from_python(pybind11::object&& data_table);

    /**
     * @brief Create RawPacketMessage cpp object from a cpp object, used internally by `create_from_cpp`
     *
     * @param data_table
     * @param index_col_count
     * @return std::shared_ptr<RawPacketMessage>
     */
    static std::shared_ptr<RawPacketMessage> create_from_cpp(uint32_t num, uint32_t max_size, uintptr_t *ptr_addr,
                                                            uint32_t *ptr_hdr_size,
                                                            uint32_t *ptr_pld_size,
                                                            bool gpu_mem,
                                                            uint16_t queue_idx = 0xFFFF);

  protected:
    RawPacketMessage(uint32_t num, uint32_t max_size, uintptr_t *ptr_addr,
                    uint32_t *ptr_hdr_size,
                    uint32_t *ptr_pld_size,
                    bool gpu_mem,
                    int queue_idx);

    /**
     * @brief Create RawPacketMessage python object from a cpp object
     *
     * @param table
     * @param index_col_count
     * @return pybind11::object
     */
    // static pybind11::object cpp_to_py(uint32_t num, uint32_t max_size, uintptr_t *ptr_addr, uint32_t *ptr_size, bool gpu_mem, uint16_t queue_idx = 0xFFFF);

    uint32_t num;
    uint32_t max_size;
    uintptr_t *ptr_addr;
    uint32_t *ptr_hdr_size;
    uint32_t *ptr_pld_size;
    uint16_t queue_idx;
  
    // std::shared_ptr<uintptr_t> m_data;
};

struct RawPacketMessageProxy
{

};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
