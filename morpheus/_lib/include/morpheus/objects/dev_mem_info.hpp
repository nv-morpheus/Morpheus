/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/utilities/type_util_detail.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** DevMemInfo******************************************/
/**
 * @brief Simple object that just holds 4 things: element count, element dtype, device_buffer, and bytes_offset
 */
struct DevMemInfo
{
    // Number of elements in the buffer
    size_t element_count;
    // Type of elements in the buffer
    neo::TypeId type_id;
    // Buffer of data
    std::shared_ptr<rmm::device_buffer> buffer;
    // Offset from head of data in bytes
    size_t offset;

    /**
     * TODO(Documentation)
     */
    void *data() const;
};

}  // namespace morpheus
