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

#include "pycoro/pycoro.hpp"

#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

#include <stdexcept>

namespace morpheus::tests::pycoro {

mrc::coroutines::Task<int> subtract(int a, int b)
{
    co_return a - b;
}

mrc::coroutines::Task<mrc::pymrc::PyHolder> call_fib_async(mrc::pymrc::PyHolder fib, int value, int minus)
{
    auto result = co_await subtract(value, minus);
    co_return co_await mrc::pycoro::PyTaskToCppAwaitable([fib, result]() {
        pybind11::gil_scoped_acquire acquire;
        return fib(result);
    }());
}

mrc::coroutines::Task<mrc::pymrc::PyHolder> raise_at_depth_async(mrc::pymrc::PyHolder fn, int depth)
{
    if (depth <= 0)
    {
        throw std::runtime_error("depth reached zero in c++");
    }

    co_return co_await mrc::pycoro::PyTaskToCppAwaitable([fn, depth]() {
        pybind11::gil_scoped_acquire acquire;
        return fn(depth - 1);
    }());
}

mrc::coroutines::Task<mrc::pymrc::PyHolder> call_async(mrc::pymrc::PyHolder fn)
{
    co_return co_await mrc::pycoro::PyTaskToCppAwaitable([fn]() {
        pybind11::gil_scoped_acquire acquire;
        return fn();
    }());
}

PYBIND11_MODULE(pycoro, _module)
{
    _module.def("call_fib_async", &call_fib_async);
    _module.def("raise_at_depth_async", &raise_at_depth_async);
    _module.def("call_async", &call_async);
}

}  // namespace morpheus::tests::pycoro
