/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/utilities/json_types.hpp"

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utils.hpp>

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<nlohmann::json>
{
  public:
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(nlohmann::json, _("object"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src || src.is_none())
        {
            return false;
        }

        value = mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src));

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

template <>
struct type_caster<nlohmann::json_dict>
{
  public:
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(nlohmann::json_dict, _("dict[str, typing.Any]"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src || src.is_none())
        {
            return false;
        }

        if (!PyDict_Check(src.ptr()))
        {
            return false;
        }

        value = mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src));

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json_dict src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

template <>
struct type_caster<nlohmann::json_list>
{
  public:
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(nlohmann::json_list, _("list[typing.Any]"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src || src.is_none())
        {
            return false;
        }

        if (!PyList_Check(src.ptr()))
        {
            return false;
        }

        value = mrc::pymrc::cast_from_pyobject(pybind11::reinterpret_borrow<pybind11::object>(src));

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(nlohmann::json_list src, return_value_policy policy, handle parent)
    {
        return mrc::pymrc::cast_from_json(src).release();
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
