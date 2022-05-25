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

#include <morpheus/messages/meta.hpp>
#include <morpheus/objects/table_info.hpp>
#include <morpheus/objects/tensor.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)
class MultiMessage
{
  public:
    MultiMessage(std::shared_ptr<MessageMeta> m, size_t o, size_t c);

    std::shared_ptr<MessageMeta> meta;
    size_t mess_offset{0};
    size_t mess_count{0};

    /**
     * TODO(Documentation)
     */
    TableInfo get_meta();

    /**
     * TODO(Documentation)
     */
    TableInfo get_meta(const std::string &col_name);

    /**
     * TODO(Documentation)
     */
    TableInfo get_meta(const std::vector<std::string> &column_names);

    /**
     * TODO(Documentation)
     */
    void set_meta(const std::string &col_name, TensorObject tensor);

    /**
     * TODO(Documentation)
     */
    void set_meta(const std::vector<std::string> &column_names, const std::vector<TensorObject> &tensors);

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MultiMessage> get_slice(size_t start, size_t stop) const;

  protected:
    // This internal function is used to allow virtual overriding while `get_slice` allows for hiding of base class.
    // This allows users to avoid casting every class after calling get_slice but still supports calling `get_slice`
    // from a base class. For example, the following all works:
    // std::shared_ptr<DerivedMultiMessage> derived_message = std::make_shared<DerivedMultiMessage>();
    //
    // // No cast is necessary here
    // std::shared_ptr<DerivedMultiMessage> other_derived = derived_message->get_slice(0, 10);
    //
    // // Conversion to base class
    // std::shared_ptr<MultiMessage> base_message = derived_message;
    //
    // // This also works
    // std::shared_ptr<MultiMessage> other_base = base_message->get_slice(0, 10);
    //
    // These will be logically equivalent
    // assert(std::dynamic_ptr_cast<DerivedMultiMessage>(other_base) == other_derived);

    /**
     * TODO(Documentation)
     */
    virtual std::shared_ptr<MultiMessage> internal_get_slice(size_t start, size_t stop) const;
};

/****** MultiMessageInterfaceProxy**************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiMessageInterfaceProxy
{
    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MultiMessage> init(std::shared_ptr<MessageMeta> meta,
                                              cudf::size_type mess_offset,
                                              cudf::size_type mess_count);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MessageMeta> meta(const MultiMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t mess_offset(const MultiMessage &self);

    /**
     * TODO(Documentation)
     */
    static std::size_t mess_count(const MultiMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_meta(MultiMessage &self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_meta(MultiMessage &self, std::string col_name);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_meta(MultiMessage &self, std::vector<std::string> columns);

    /**
     * TODO(Documentation)
     * @note I think this was a bug, we have two overloads with the same function signatures
     */
    static pybind11::object get_meta_by_col(MultiMessage &self, pybind11::object columns);

    /**
     * TODO(Documentation)
     */
    static void set_meta(MultiMessage &self, pybind11::object columns, pybind11::object value);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MultiMessage> get_slice(MultiMessage &self, std::size_t start, std::size_t stop);
};

#pragma GCC visibility pop
}  // namespace morpheus
