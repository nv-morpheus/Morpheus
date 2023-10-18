/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Original Source: https://github.com/lewissbaker/cppcoro
 * Original License: MIT; included below
 */

///////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lewis Baker
// Licenced under MIT license. See LICENSE.txt for details.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "async_generator.hpp"

#include <boost/type_traits/remove_reference.hpp>
#include <mrc/coroutines/concepts/awaitable.hpp>
#include <mrc/coroutines/task.hpp>

#include <type_traits>

namespace mrc::coroutines {

template <typename SchedulerT, typename AwaitableT>
auto schedule_on(SchedulerT& scheduler, AwaitableT awaitable) -> Task<boost::detail::remove_rvalue_ref<
    typename mrc::coroutines::concepts::awaitable_traits<AwaitableT>::awaiter_return_type>>
{
    co_await scheduler.schedule();
    co_return co_await std::move(awaitable);
}

template <typename T, typename SchedulerT>
mrc::coroutines::AsyncGenerator<T> schedule_on(SchedulerT& scheduler, mrc::coroutines::AsyncGenerator<T> source)
{
    // Transfer exection to the scheduler before the implicit calls to
    // 'co_await begin()' or subsequent calls to `co_await iterator::operator++()`
    // below. This ensures that all calls to the generator's coroutine_handle<>::resume()
    // are executed on the execution context of the scheduler.
    co_await scheduler.schedule();

    const auto itEnd = source.end();
    auto it          = co_await source.begin();
    while (it != itEnd)
    {
        co_yield *it;

        co_await scheduler.schedule();

        (void)co_await ++it;
    }
}

}  // namespace mrc::coroutines
