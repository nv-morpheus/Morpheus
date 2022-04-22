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

#include <morpheus/objects/tensor.hpp>

#include <neo/core/tensor.hpp>

#include <cstddef>
#include <map>
#include <string>

namespace morpheus {

    /**
     * TODO(Documentation)
     */
    class InferenceMemory {
    public:
        InferenceMemory(size_t count);

        std::size_t count{0};
        std::map<std::string, neo::TensorObject> inputs;

        /**
         * TODO(Documentation)
         */
        bool has_input(const std::string &name) const;
    };

    /****** InferenceMemoryInterfaceProxy *************************/
#pragma GCC visibility push(default)
    /**
     * @brief Interface proxy, used to insulate python bindings.
     */
    struct InferenceMemoryInterfaceProxy {
        /**
         * TODO(Documentation)
         */
        static std::size_t get_count(InferenceMemory& self);
    };
#pragma GCC visibility pop
}