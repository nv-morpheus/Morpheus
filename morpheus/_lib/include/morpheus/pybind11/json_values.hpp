/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utilities/json_values.hpp>

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

template <>
struct type_caster<mrc::pymrc::JSONValues>
{
  public:
    /**
     * This macro establishes a local variable 'value' of type JSONValues
     */
    PYBIND11_TYPE_CASTER(mrc::pymrc::JSONValues, _("object"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into JSONValues
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src)
        {
            return false;
        }

        if (src.is_none())
        {
            value = mrc::pymrc::JSONValues();
        }
        else
        {
            value = std::move(mrc::pymrc::JSONValues(pybind11::reinterpret_borrow<pybind11::object>(src)));
        }

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert a JSONValues instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(mrc::pymrc::JSONValues src, return_value_policy policy, handle parent)
    {
        return src.to_python().release();
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
