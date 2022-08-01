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
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"

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
#include <utility>  // for pair
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiMessage****************************************/
/**
 * TODO(Documentation)
 */
#pragma GCC visibility push(default)

class MultiMessage;

/**
 * @brief All classes that are derived from MultiMessage should use this class. It will automatically add the
 * `get_slice` function with the correct return type. Uses the CRTP pattern. This supports multiple base classes but in
 * reality it should not be used. Multiple base classes are used to make template specialization easier.
 *
 * @tparam DerivedT The deriving class. Should be used like `class MyDerivedMultiMessage: public
 *         DerivedMultiMessage<MyDerivedMultiMessage, MultiMessage>`
 * @tparam BasesT The base classes that the class should derive from. If the class should function like `class
 *         MyDerivedMultiMessage: public MyBaseMultiMessage`, then `class MyDerivedMultiMessage: public
 *         DerivedMultiMessage<MyDerivedMultiMessage, MyBaseMultiMessage>` shoud be used.
 */
template <typename DerivedT, typename... BasesT>
class DerivedMultiMessage : public BasesT...
{
  public:
    virtual ~DerivedMultiMessage() = default;

    /**
     * @brief Creates a copy of the current message calculating new `mess_offset` and `mess_count` values based on the
     * given `start` & `stop` values. This method is reletively light-weight as it does not copy the underlying `meta`
     * and the actual slicing of the dataframe is applied later when `get_meta` is called.
     *
     * @param start
     * @param stop
     * @return std::shared_ptr<DerivedT>
     */
    std::shared_ptr<DerivedT> get_slice(std::size_t start, std::size_t stop) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->get_slice_impl(new_message, start, stop);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

    /**
     * @brief Creates a deep copy of the current message along with a copy of the underlying `meta` selecting the rows
     * of `meta` defined by pairs of start, stop rows expressed in the `ranges` argument.
     *
     * This allows for copying several non-contiguous rows from the underlying dataframe into a new dataframe, however
     * this comes at a much higher cost compared to the `get_slice` method.
     *
     * @param ranges
     * @param num_selected_rows
     * @return std::shared_ptr<DerivedT>
     */
    std::shared_ptr<DerivedT> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                          size_t num_selected_rows) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->copy_ranges_impl(new_message, ranges, num_selected_rows);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

  protected:
    /**
     * @brief Applies a slice of the attribures contained in `new_message`. Subclasses need only be concerned with their
     * own attributes, and can safely avoid overriding this method if they don't add any new attributes to their base.
     *
     * @param new_message
     * @param start
     * @param stop
     */
    virtual void get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                std::size_t start,
                                std::size_t stop) const = 0;

    /**
     * @brief Similar to `get_slice_impl`, performs a copy of all attributes in `new_message` according to the rows
     * specified by `ranges`. Subclasses need only be concerned with copying their own attributes, and can safely avoid
     * overriding this method if they don't add any new attributes to their base.
     *
     * @param new_message
     * @param ranges
     * @param num_selected_rows
     */
    virtual void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                  const std::vector<std::pair<size_t, size_t>> &ranges,
                                  size_t num_selected_rows) const = 0;

  private:
    virtual std::shared_ptr<MultiMessage> clone_impl() const
    {
        // Cast `this` to the derived type
        auto derived_this = static_cast<const DerivedT *>(this);

        // Use copy constructor to make a clone
        return std::make_shared<DerivedT>(*derived_this);
    }
};

// Single base class version. Should be the version used by default
template <typename DerivedT, typename BaseT>
class DerivedMultiMessage<DerivedT, BaseT> : public BaseT
{
  public:
    using BaseT::BaseT;
    virtual ~DerivedMultiMessage() = default;

    std::shared_ptr<DerivedT> get_slice(std::size_t start, std::size_t stop) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->get_slice_impl(new_message, start, stop);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

    std::shared_ptr<DerivedT> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                          size_t num_selected_rows) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->copy_ranges_impl(new_message, ranges, num_selected_rows);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

  protected:
    virtual void get_slice_impl(std::shared_ptr<MultiMessage> new_message, std::size_t start, std::size_t stop) const
    {
        return BaseT::get_slice_impl(new_message, start, stop);
    }

    virtual void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                  const std::vector<std::pair<size_t, size_t>> &ranges,
                                  size_t num_selected_rows) const
    {
        return BaseT::copy_ranges_impl(new_message, ranges, num_selected_rows);
    }

  private:
    virtual std::shared_ptr<MultiMessage> clone_impl() const
    {
        // Cast `this` to the derived type
        auto derived_this = static_cast<const DerivedT *>(this);

        // Use copy constructor to make a clone
        return std::make_shared<DerivedT>(*derived_this);
    }
};

// No base class version. This should only be used by `MultiMessage` itself.
template <typename DerivedT>
class DerivedMultiMessage<DerivedT>
{
  public:
    virtual ~DerivedMultiMessage() = default;

    std::shared_ptr<DerivedT> get_slice(std::size_t start, std::size_t stop) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->get_slice_impl(new_message, start, stop);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

    std::shared_ptr<DerivedT> copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                          size_t num_selected_rows) const
    {
        std::shared_ptr<MultiMessage> new_message = this->clone_impl();

        this->copy_ranges_impl(new_message, ranges, num_selected_rows);

        return DCHECK_NOTNULL(std::dynamic_pointer_cast<DerivedT>(new_message));
    }

  protected:
    virtual void get_slice_impl(std::shared_ptr<MultiMessage> new_message,
                                std::size_t start,
                                std::size_t stop) const = 0;

    virtual void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                  const std::vector<std::pair<size_t, size_t>> &ranges,
                                  size_t num_selected_rows) const = 0;

  private:
    virtual std::shared_ptr<MultiMessage> clone_impl() const
    {
        // Cast `this` to the derived type
        auto derived_this = static_cast<const DerivedT *>(this);

        // Use copy constructor to make a clone
        return std::make_shared<DerivedT>(*derived_this);
    }
};

class MultiMessage : public DerivedMultiMessage<MultiMessage>
{
  public:
    MultiMessage(const MultiMessage &other) = default;
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

  protected:
    void get_slice_impl(std::shared_ptr<MultiMessage> new_message, std::size_t start, std::size_t stop) const override;

    void copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                          const std::vector<std::pair<size_t, size_t>> &ranges,
                          size_t num_selected_rows) const override;

    /**
     * @brief Creates a deep copy of `meta` with the specified ranges.
     *
     * @param ranges
     * @return std::shared_ptr<MessageMeta>
     */
    virtual std::shared_ptr<MessageMeta> copy_meta_ranges(const std::vector<std::pair<size_t, size_t>> &ranges) const;

    /**
     * @brief Applies the message offset to the elements in `ranges` casting the results to `TensorIndex`
     *
     * @param ranges
     * @return std::vector<std::pair<TensorIndex, TensorIndex>>
     */
    std::vector<std::pair<TensorIndex, TensorIndex>> apply_offset_to_ranges(
        std::size_t offset, const std::vector<std::pair<size_t, size_t>> &ranges) const;
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

    static pybind11::object get_meta_list(MultiMessage &self, pybind11::object col_name);

    /**
     * TODO(Documentation)
     */
    static void set_meta(MultiMessage &self, pybind11::object columns, pybind11::object value);

    /**
     * TODO(Documentation)
     */
    static std::shared_ptr<MultiMessage> get_slice(MultiMessage &self, std::size_t start, std::size_t stop);

    static std::shared_ptr<MultiMessage> copy_ranges(MultiMessage &self,
                                                     const std::vector<std::pair<size_t, size_t>> &ranges,
                                                     pybind11::object num_selected_rows);
};

#pragma GCC visibility pop
}  // namespace morpheus
