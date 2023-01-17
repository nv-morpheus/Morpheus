/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/wrapped_tensor.hpp"

#include "morpheus/objects/tensor_object.hpp"

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <array>    // needed for make_tuple
#include <cstdint>  // for uintptr_t
#include <vector>   // get_shape & get_stride return vectors

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorObject****************************************/
/****** TensorObjectInterfaceProxy *************************/
pybind11::dict TensorObjectInterfaceProxy::cuda_array_interface(TensorObject &self)
{
    pybind11::dict array_interface;

    pybind11::list shape_list;

    for (auto &idx : self.get_shape())
    {
        shape_list.append(idx);
    }

    pybind11::list stride_list;

    for (auto &idx : self.get_stride())
    {
        stride_list.append(idx * self.dtype_size());
    }

    // pybind11::list shape_list = pybind11::cast(self.get_shape());

    pybind11::int_ stream_val = 1;

    // See https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    // if (self.get_stream().is_default())
    // {
    //     stream_val = 1;
    // }
    // else if (self.get_stream().is_per_thread_default())
    // {
    //     stream_val = 2;
    // }
    // else
    // {
    //     // Custom stream. Return value
    //     stream_val = (int64_t)self.get_stream().value();
    // }

    array_interface["shape"]   = pybind11::cast<pybind11::tuple>(shape_list);
    array_interface["typestr"] = self.get_numpy_typestr();
    array_interface["stream"]  = stream_val;
    array_interface["version"] = 3;

    if (self.is_compact() || self.get_stride().empty())
    {
        array_interface["strides"] = pybind11::none();
    }
    else
    {
        array_interface["strides"] = pybind11::cast<pybind11::tuple>(stride_list);
    }
    array_interface["data"] = pybind11::make_tuple((uintptr_t)self.data(), false);

    return array_interface;
}
}  // namespace morpheus
