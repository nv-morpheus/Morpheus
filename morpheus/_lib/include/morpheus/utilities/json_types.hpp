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

// namespace py = pybind11;
// using namespace py::literals;

namespace morpheus::utilities {
/**
 * @brief A container class derived from std::vector<uint8_t> to make it compatible with nlohmann::json to hold
 * arbitrary Python objects as bytes.
 *
 */
class MORPHEUS_EXPORT PythonByteContainer : public std::vector<uint8_t>
{
  public:
    /**
     * @brief Construct a new Python Byte Container object
     *
     */
    PythonByteContainer() = default;

    /**
     * @brief Construct a new Python Byte Container object by initializing it with a `mrc::pymrc::PyHolder`.
     *
     * @param py_obj a PyHolder object that holds a Python object to be stored into the container
     */
    PythonByteContainer(mrc::pymrc::PyHolder py_obj);

    /**
     * @brief Get the PyHolder object from the container
     *
     * @return mrc::pymrc::PyHolder the PyHolder object stored in the container
     */
    mrc::pymrc::PyHolder get_py_obj() const;

  private:
    mrc::pymrc::PyHolder m_py_obj;
};

/**
 * @brief  * A specialization of `nlohmann::basic_json` with customized BinaryType `PythonByteContainer` to hold Python
 * objects as bytes.
 *
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

/**
 * @brief Convert a `json_t` object to a pybind11 object. The difference to `mrc::pymrc::cast_from_json()` is that if
 * the object cannot be serialized, it checks if the object contains a supported binary type. Otherwise,
 * pybind11::none is returned.
 *
 * @param source : `json_t` object
 * @return pybind11 object
 */
MORPHEUS_EXPORT pybind11::object cast_from_json(const morpheus::utilities::json_t& source);

/**
 * @brief Convert a pybind11 object to a json_t object. The difference to `mrc::pymrc::cast_from_pyobject` is that if
 * the object cannot be serialized, it wraps the python object in a `PythonByteContainer` and returns it as a binary.
 *
 * @param source : pybind11 object
 * @return json_t object.
 */
MORPHEUS_EXPORT json_t cast_from_pyobject(const pybind11::object& source);

// NOLINTBEGIN(readability-identifier-naming)
/*
    Derived class from json_t to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
class MORPHEUS_EXPORT json_dict_t : public morpheus::utilities::json_t
{};

/*
    Derived class from json_t to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
class MORPHEUS_EXPORT json_list_t : public morpheus::utilities::json_t
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
