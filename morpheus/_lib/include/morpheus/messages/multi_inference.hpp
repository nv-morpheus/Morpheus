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

#include <morpheus/messages/memory/inference_memory.hpp>
#include <morpheus/messages/meta.hpp>
#include <morpheus/messages/multi.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/objects/tensor_object.hpp>

#include <pybind11/pytypes.h>
#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations********************/
/****** MultiInferenceMessage*******************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiInferenceMessage : public MultiMessage
{
  public:
    MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                          std::size_t mess_offset,
                          std::size_t mess_count,
                          std::shared_ptr<morpheus::InferenceMemory> memory,
                          std::size_t offset,
                          std::size_t count);

    std::shared_ptr<morpheus::InferenceMemory> memory;
    std::size_t offset{0};
    std::size_t count{0};

    /**
     * TODO(Documentation)
     */
    const TensorObject get_input(const std::string &name) const;

    /**
     * TODO(Documentation)
     */
    const void set_input(const std::string &name, const TensorObject &value);

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MultiInferenceMessage> get_slice(std::size_t start, std::size_t stop) const;

    std::shared_ptr<MultiInferenceMessage> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                       size_t num_selected_rows) const;

  protected:
    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MultiMessage> internal_get_slice(std::size_t start, std::size_t stop) const override;

    std::shared_ptr<MultiMessage> internal_copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                       size_t num_selected_rows) const override;

    std::shared_ptr<InferenceMemory> copy_input_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                       size_t num_selected_rows) const;
};

/****** MultiInferenceMessageInterfaceProxy****************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiInferenceMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiInferenceMessage object, and return a shared pointer to the result.
     */
    static std::shared_ptr<MultiInferenceMessage> init(std::shared_ptr<MessageMeta> meta,
                                                       cudf::size_type mess_offset,
                                                       cudf::size_type mess_count,
                                                       std::shared_ptr<InferenceMemory> memory,
                                                       cudf::size_type offset,
                                                       cudf::size_type count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<morpheus::InferenceMemory> memory(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t offset(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(MultiInferenceMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_input(MultiInferenceMessage &self, const std::string &name);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MultiInferenceMessage> get_slice(MultiInferenceMessage &self,
                                                            std::size_t start,
                                                            std::size_t stop);
};
#pragma GCC visibility pop
}  // namespace morpheus
