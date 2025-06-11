/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/tensor_object.hpp"

namespace morpheus {

static void set_contiguous_stride(const ShapeType& shape, ShapeType& stride)
{
    stride.resize(shape.size());
    TensorIndex ttl = 1;
    auto rank       = shape.size();
    for (int i = rank - 1; i >= 0; i--)
    {
        stride[i] = ttl;
        ttl *= shape.at(i);
    }
}

void TensorObject::throw_on_invalid_storage() {}

}  // namespace morpheus
