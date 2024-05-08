/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"
#include <nlohmann/json.hpp>
#include <pymrc/types.hpp>


namespace py = pybind11;
using namespace py::literals;

namespace morpheus::utilities {
class MORPHEUS_EXPORT PythonByteContainer : public std::vector<uint8_t>
{
  public:
    PythonByteContainer() = default;
    PythonByteContainer(mrc::pymrc::PyHolder py_obj);

    mrc::pymrc::PyHolder get_py_obj() const;

  private:
    mrc::pymrc::PyHolder m_py_obj;
};

/**
 * A specialization of nlohmann::basic_json with customized BinaryType (PythonByteContainer) to hold Python objects
 * as bytes.
 */
using json_t = nlohmann::basic_json<std::map,
                                    std::vector,
                                    std::string,
                                    bool,
                                    std::int64_t,
                                    std::uint64_t,
                                    double,
                                    std::allocator,
                                    nlohmann::adl_serializer,
                                    PythonByteContainer,
                                    void>;

MORPHEUS_EXPORT py::object cast_from_json(const morpheus::utilities::json_t& source);
MORPHEUS_EXPORT json_t cast_from_pyobject(const py::object& source, mrc::pymrc::unserializable_handler_fn_t unserializable_handler_fn);
MORPHEUS_EXPORT json_t cast_from_pyobject(const py::object& source);

// NOLINTBEGIN(readability-identifier-naming)
/*
    Derived class from json_t to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
class MORPHEUS_EXPORT json_t_dict : public morpheus::utilities::json_t
{};

/*
    Derived class from json_t to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
class MORPHEUS_EXPORT json_t_list : public morpheus::utilities::json_t
{};
// NOLINTEND(readability-identifier-naming)
}  // namespace morpheus::utilities

namespace nlohmann {
// NOLINTBEGIN(readability-identifier-naming)

/*
    Derived class from basic_json to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
// NLOHMANN_BASIC_JSON_TPL_DECLARATION
class json_dict : public basic_json<>
{};

/*
    Derived class from basic_json to allow for custom type names. Use this if the return type would always be a list
   (i.e. list[Any] in python)
*/
class json_list : public basic_json<>
{};

// NOLINTEND(readability-identifier-naming)
}  // namespace nlohmann
