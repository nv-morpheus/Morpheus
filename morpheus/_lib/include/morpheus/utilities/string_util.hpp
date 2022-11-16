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

#include <sstream>
#include <string>

namespace morpheus {

/**
 * @addtogroup utilities
 * @{
 * @file
*/

// Concats multiple strings together using ostringstream. Use with MORPHEUS_CONCAT_STR("Start [" << my_int << "]")
#define MORPHEUS_CONCAT_STR(strs) ((std::ostringstream&)(std::ostringstream() << strs)).str()

/****** Component public implementations *******************/
/****** StringUtil****************************************/
/**
 * @brief Structure that encapsulates string utilities.
 */
struct StringUtil
{
    /**
     * TODO(Documentation)
     */
    template <typename IterT>
    static std::string join(IterT begin, IterT end, std::string const& separator)
    {
        std::ostringstream result;
        if (begin != end)
            result << *begin++;
        while (begin != end)
            result << separator << *begin++;
        return result.str();
    }

    /**
     * TODO(Documentation)
     */
    template <typename IterT>
    static std::string array_to_str(IterT begin, IterT end)
    {
        return MORPHEUS_CONCAT_STR("[" << join(begin, end, ", ") << "]");
    }

    /**
     * TODO(Documentation)
     */
    static bool str_contains(const std::string& str, const std::string& search_str);
};
/** @} */  // end of group
}  // namespace morpheus
