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

#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/type_util_detail.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace morpheus {

/**
 * @addtogroup utilities
 * @{
 * @file
*/

struct MatxUtil
{
    /**
     * @brief Convert one device_buffer type to another
     * @return
     */
    static std::shared_ptr<rmm::device_buffer> cast(const DevMemInfo &input, TypeId output_type);

    /**
     * @brief Builds a Nx3 segment ID matrix
     * @return
     */
    static std::shared_ptr<rmm::device_buffer> create_seg_ids(size_t row_count, size_t fea_len, TypeId output_type);

    /**
     * @brief Calculate logits on device_buffer
     * @return
     */
    static std::shared_ptr<rmm::device_buffer> logits(const DevMemInfo &input);

    /**
     * @brief Perform transpose
     * @return
     */
    static std::shared_ptr<rmm::device_buffer> transpose(const DevMemInfo &input, size_t rows, size_t cols);

    /**
     * @brief Return an array of boolean where x[i,j] >= thresh_val, when by_row is true an Nx1 array will be returned
     * with a true if any value in the row is above the threshold
     * @return
     */
    static std::shared_ptr<rmm::device_buffer> threshold(const DevMemInfo &input,
                                                         size_t rows,
                                                         size_t cols,
                                                         const std::vector<TensorIndex> &stride,
                                                         double thresh_val,
                                                         bool by_row);

    /**
     * @brief Returns a buffer with `output_shape` containing the max value from values in `input` mapped according to
     * `seq_ids`.
     * Ex given a hypothetical input of:
     *
     *     input =   [5, 2, 8, 9, 8, 2, 1]
     *     seq_ids = [0, 0, 0, 1, 2, 3, 3]
     *
     * Will return:
     *               [8, 9, 8, 2]
     * @return std::shared_ptr<rmm::device_buffer>
     */
    static std::shared_ptr<rmm::device_buffer> reduce_max(const DevMemInfo &input,
                                                          const std::vector<int32_t> &seq_ids,
                                                          size_t seq_id_offset,
                                                          const std::vector<int64_t> &input_shape,
                                                          const std::vector<int64_t> &input_stride,
                                                          const std::vector<int64_t> &output_shape);
};
/** @} */  // end of group
}  // namespace morpheus
