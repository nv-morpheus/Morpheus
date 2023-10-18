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

// Copyright 2017 Lewis Baker

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <concepts>
#include <coroutine>
#include <exception>
#include <type_traits>

namespace mrc::coroutines {

template <typename T>
class AsyncGenerator;

namespace detail {

template <typename T>
class AsyncGeneratorIterator;
class AsyncGeneratorYieldOperation;
class AsyncGeneratorAdvanceOperation;

class AsyncGeneratorPromiseBase
{
  public:
    AsyncGeneratorPromiseBase() noexcept : m_exception(nullptr) {}

    DELETE_COPYABILITY(AsyncGeneratorPromiseBase)

    constexpr static std::suspend_always initial_suspend() noexcept
    {
        return {};
    }

    AsyncGeneratorYieldOperation final_suspend() noexcept;

    void unhandled_exception() noexcept
    {
        m_exception = std::current_exception();
    }

    auto return_void() noexcept -> void {}

    auto finished() const noexcept -> bool
    {
        return m_value == nullptr;
    }

    auto rethrow_on_unhandled_exception() -> void
    {
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }
    }

  protected:
    AsyncGeneratorYieldOperation internal_yield_value() noexcept;
    void* m_value{nullptr};

  private:
    std::exception_ptr m_exception;
    std::coroutine_handle<> m_consumer;

    friend class AsyncGeneratorYieldOperation;
    friend class AsyncGeneratorAdvanceOperation;
};

class AsyncGeneratorYieldOperation final
{
  public:
    AsyncGeneratorYieldOperation(std::coroutine_handle<> consumer) noexcept : m_consumer(consumer) {}

    constexpr static bool await_ready() noexcept
    {
        return false;
    }

    std::coroutine_handle<> await_suspend([[maybe_unused]] std::coroutine_handle<> producer) const noexcept
    {
        return m_consumer;
    }

    constexpr static void await_resume() noexcept {}

  private:
    std::coroutine_handle<> m_consumer;
};

inline AsyncGeneratorYieldOperation AsyncGeneratorPromiseBase::final_suspend() noexcept
{
    m_value = nullptr;
    return internal_yield_value();
}

inline AsyncGeneratorYieldOperation AsyncGeneratorPromiseBase::internal_yield_value() noexcept
{
    return AsyncGeneratorYieldOperation{m_consumer};
}

class AsyncGeneratorAdvanceOperation
{
  protected:
    AsyncGeneratorAdvanceOperation(std::nullptr_t) noexcept : m_promise(nullptr), m_producer(nullptr) {}

    AsyncGeneratorAdvanceOperation(AsyncGeneratorPromiseBase& promise, std::coroutine_handle<> producer) noexcept :
      m_promise(std::addressof(promise)),
      m_producer(producer)
    {}

  public:
    constexpr static bool await_ready() noexcept
    {
        return false;
    }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> consumer) noexcept
    {
        m_promise->m_consumer = consumer;
        return m_producer;
    }

  protected:
    AsyncGeneratorPromiseBase* m_promise;
    std::coroutine_handle<> m_producer;
};

template <typename T>
class AsyncGeneratorPromise final : public AsyncGeneratorPromiseBase
{
    using value_t     = std::remove_reference_t<T>;
    using reference_t = std::conditional_t<std::is_reference_v<T>, T, T&>;
    using pointer_t   = value_t*;

  public:
    AsyncGeneratorPromise() noexcept = default;

    AsyncGenerator<T> get_return_object() noexcept;

    template <typename U = T, std::enable_if_t<!std::is_rvalue_reference<U>::value, int> = 0>
    auto yield_value(value_t& value) noexcept -> AsyncGeneratorYieldOperation
    {
        m_value = std::addressof(value);
        return internal_yield_value();
    }

    auto yield_value(value_t&& value) noexcept -> AsyncGeneratorYieldOperation
    {
        m_value = std::addressof(value);
        return internal_yield_value();
    }

    auto value() const noexcept -> reference_t
    {
        return *static_cast<pointer_t>(m_value);
    }
};

template <typename T>
class AsyncGeneratorIncrementOperation final : public AsyncGeneratorAdvanceOperation
{
  public:
    AsyncGeneratorIncrementOperation(AsyncGeneratorIterator<T>& iterator) noexcept :
      AsyncGeneratorAdvanceOperation(iterator.m_coroutine.promise(), iterator.m_coroutine),
      m_iterator(iterator)
    {}

    AsyncGeneratorIterator<T>& await_resume();

  private:
    AsyncGeneratorIterator<T>& m_iterator;
};

struct AsyncGeneratorSentinel
{};

template <typename T>
class AsyncGeneratorIterator final
{
    using promise_t = AsyncGeneratorPromise<T>;
    using handle_t  = std::coroutine_handle<promise_t>;

  public:
    using iterator_category = std::input_iterator_tag;  // NOLINT
    // Not sure what type should be used for difference_type as we don't
    // allow calculating difference between two iterators.
    using difference_t = std::ptrdiff_t;
    using value_t      = std::remove_reference_t<T>;
    using reference    = std::add_lvalue_reference_t<T>;  // NOLINT
    using pointer      = std::add_pointer_t<value_t>;     // NOLINT

    AsyncGeneratorIterator(std::nullptr_t) noexcept : m_coroutine(nullptr) {}

    AsyncGeneratorIterator(handle_t coroutine) noexcept : m_coroutine(coroutine) {}

    AsyncGeneratorIncrementOperation<T> operator++() noexcept
    {
        return AsyncGeneratorIncrementOperation<T>{*this};
    }

    reference operator*() const noexcept
    {
        return m_coroutine.promise().value();
    }

    bool operator==(const AsyncGeneratorIterator& other) const noexcept
    {
        return m_coroutine == other.m_coroutine;
    }

    bool operator!=(const AsyncGeneratorIterator& other) const noexcept
    {
        return !(*this == other);
    }

    operator bool() const noexcept
    {
        return m_coroutine && !m_coroutine.promise().finished();
    }

  private:
    friend class AsyncGeneratorIncrementOperation<T>;

    handle_t m_coroutine;
};

template <typename T>
inline AsyncGeneratorIterator<T>& AsyncGeneratorIncrementOperation<T>::await_resume()
{
    if (m_promise->finished())
    {
        // Update iterator to end()
        m_iterator = AsyncGeneratorIterator<T>{nullptr};
        m_promise->rethrow_on_unhandled_exception();
    }

    return m_iterator;
}

template <typename T>
class AsyncGeneratorBeginOperation final : public AsyncGeneratorAdvanceOperation
{
    using promise_t = AsyncGeneratorPromise<T>;
    using handle_t  = std::coroutine_handle<promise_t>;

  public:
    AsyncGeneratorBeginOperation(std::nullptr_t) noexcept : AsyncGeneratorAdvanceOperation(nullptr) {}

    AsyncGeneratorBeginOperation(handle_t producer) noexcept :
      AsyncGeneratorAdvanceOperation(producer.promise(), producer)
    {}

    bool await_ready() const noexcept
    {
        return m_promise == nullptr || AsyncGeneratorAdvanceOperation::await_ready();
    }

    AsyncGeneratorIterator<T> await_resume()
    {
        if (m_promise == nullptr)
        {
            // Called begin() on the empty generator.
            return AsyncGeneratorIterator<T>{nullptr};
        }

        if (m_promise->finished())
        {
            // Completed without yielding any values.
            m_promise->rethrow_on_unhandled_exception();
            return AsyncGeneratorIterator<T>{nullptr};
        }

        return AsyncGeneratorIterator<T>{handle_t::from_promise(*static_cast<promise_t*>(m_promise))};
    }
};

}  // namespace detail

template <typename T>
class [[nodiscard]] AsyncGenerator
{
  public:
    // There must be a type called `promise_type` for coroutines to work. Skil linting
    using promise_type = detail::AsyncGeneratorPromise<T>;   // NOLINT(readability-identifier-naming)
    using iterator     = detail::AsyncGeneratorIterator<T>;  // NOLINT(readability-identifier-naming)

    AsyncGenerator() noexcept : m_coroutine(nullptr) {}

    explicit AsyncGenerator(promise_type& promise) noexcept :
      m_coroutine(std::coroutine_handle<promise_type>::from_promise(promise))
    {}

    AsyncGenerator(AsyncGenerator&& other) noexcept : m_coroutine(other.m_coroutine)
    {
        other.m_coroutine = nullptr;
    }

    ~AsyncGenerator()
    {
        if (m_coroutine)
        {
            m_coroutine.destroy();
        }
    }

    AsyncGenerator& operator=(AsyncGenerator&& other) noexcept
    {
        AsyncGenerator temp(std::move(other));
        swap(temp);
        return *this;
    }

    AsyncGenerator(const AsyncGenerator&)            = delete;
    AsyncGenerator& operator=(const AsyncGenerator&) = delete;

    auto begin() noexcept
    {
        if (!m_coroutine)
        {
            return detail::AsyncGeneratorBeginOperation<T>{nullptr};
        }

        return detail::AsyncGeneratorBeginOperation<T>{m_coroutine};
    }

    auto end() noexcept
    {
        return iterator{nullptr};
    }

    void swap(AsyncGenerator& other) noexcept
    {
        using std::swap;
        swap(m_coroutine, other.m_coroutine);
    }

  private:
    std::coroutine_handle<promise_type> m_coroutine;
};

template <typename T>
void swap(AsyncGenerator<T>& a, AsyncGenerator<T>& b) noexcept
{
    a.swap(b);
}

namespace detail {
template <typename T>
AsyncGenerator<T> AsyncGeneratorPromise<T>::get_return_object() noexcept
{
    return AsyncGenerator<T>{*this};
}

}  // namespace detail

}  // namespace mrc::coroutines
