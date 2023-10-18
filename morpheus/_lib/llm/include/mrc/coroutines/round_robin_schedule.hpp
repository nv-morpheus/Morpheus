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

#include <algorithm>
#include <array>
#include <cassert>
#include <coroutine>
#include <utility>

namespace mrc::coroutine {
/// This is a scheduler class that schedules coroutines in a round-robin
/// fashion once N coroutines have been scheduled to it.
///
/// Only supports access from a single thread at a time so
///
/// This implementation was inspired by Gor Nishanov's CppCon 2018 talk
/// about nano-coroutines.
///
/// The implementation relies on symmetric transfer and noop_coroutine()
/// and so only works with a relatively recent version of Clang and does
/// not yet work with MSVC.
template <size_t N>
class RoundRobinScheduler
{
    static_assert(N >= 2, "Round robin scheduler must be configured to support at least two coroutines");

    class ScheduleOperation
    {
      public:
        explicit ScheduleOperation(RoundRobinScheduler& s) noexcept : m_scheduler(s) {}

        bool await_ready() noexcept
        {
            return false;
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaitingCoroutine) noexcept
        {
            return m_scheduler.exchange_next(awaitingCoroutine);
        }

        void await_resume() noexcept {}

      private:
        RoundRobinScheduler& m_scheduler;
    };

    friend class ScheduleOperation;

  public:
    RoundRobinScheduler() noexcept : m_noop(std::noop_coroutine())
    {
        for (size_t i = 0; i < N - 1; ++i)
        {
            m_coroutines[i] = m_noop();
        }
    }

    ~RoundRobinScheduler()
    {
        // All tasks should have been joined before calling destructor.
        assert(std::all_of(m_coroutines.begin(), m_coroutines.end(), [&](auto h) {
            return h == m_noop;
        }));
    }

    ScheduleOperation schedule() noexcept
    {
        return ScheduleOperation{*this};
    }

    /// Resume any queued coroutines until there are no more coroutines.
    void drain() noexcept
    {
        size_t countRemaining = N - 1;
        do
        {
            auto nextToResume = exchange_next(m_noop);
            if (nextToResume != m_noop)
            {
                nextToResume.resume();
                countRemaining = N - 1;
            }
            else
            {
                --countRemaining;
            }
        } while (countRemaining > 0);
    }

  private:
    std::coroutine_handle<> exchange_next(std::coroutine_handle<> coroutine) noexcept
    {
        auto coroutineToResume = std::exchange(m_coroutines[m_index], coroutine);

        m_index = m_index < (N - 2) ? m_index + 1 : 0;

        return coroutineToResume;
    }

    size_t m_index{};
    const std::noop_coroutine_handle m_noop;
    std::array<std::coroutine_handle<>, N - 1> m_coroutines;
};

}  // namespace mrc::coroutine
