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

#pragma once

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/descr.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

#include <coroutine>
#include <exception>
#include <memory>
#include <ostream>
#include <utility>

// Dont directly include python headers
// IWYU pragma: no_include <genobject.h>

namespace mrc::pycoro {

class PYBIND11_EXPORT StopIteration : public pybind11::stop_iteration
{
  public:
    StopIteration(pybind11::object&& result) : stop_iteration("--"), m_result(std::move(result)){};
    ~StopIteration() override;

    void set_error() const override
    {
        PyErr_SetObject(PyExc_StopIteration, this->m_result.ptr());
    }

  private:
    pybind11::object m_result;
};

class PYBIND11_EXPORT CppToPyAwaitable : public std::enable_shared_from_this<CppToPyAwaitable>
{
  public:
    CppToPyAwaitable() = default;

    template <typename T>
    CppToPyAwaitable(mrc::coroutines::Task<T>&& task)
    {
        auto converter = [](mrc::coroutines::Task<T> incoming_task) -> mrc::coroutines::Task<mrc::pymrc::PyHolder> {
            DCHECK_EQ(PyGILState_Check(), 0) << "Should not have the GIL when resuming a C++ coroutine";

            mrc::pymrc::PyHolder holder;

            if constexpr (std::is_same_v<void, T>)
            {
                co_await incoming_task;

                // Need the GIL to make the return object
                pybind11::gil_scoped_acquire gil;

                holder = pybind11::none();
            }
            else
            {
                auto result = co_await incoming_task;

                // Need the GIL to cast the return object
                pybind11::gil_scoped_acquire gil;

                holder = pybind11::cast(std::move(result));
            }

            co_return holder;
        };

        m_task = converter(std::move(task));
    }

    CppToPyAwaitable(mrc::coroutines::Task<mrc::pymrc::PyHolder>&& task) : m_task(std::move(task)) {}

    std::shared_ptr<CppToPyAwaitable> iter()
    {
        return this->shared_from_this();
    }

    std::shared_ptr<CppToPyAwaitable> await()
    {
        return this->shared_from_this();
    }

    void next()
    {
        // Need to release the GIL before waiting
        pybind11::gil_scoped_release nogil;

        // Run the tick function which will resume the coroutine
        this->tick();

        if (m_task.is_ready())
        {
            // job done -> throw
            auto exception = StopIteration(std::move(m_task.promise().result()));

            // Destroy the task now that we have the value
            m_task.destroy();

            throw exception;
        }
    }

  protected:
    virtual void tick()
    {
        if (!m_has_resumed)
        {
            m_has_resumed = true;

            pybind11::gil_scoped_release nogil;

            m_task.resume();
        }
    }

    bool m_has_resumed{false};
    mrc::coroutines::Task<mrc::pymrc::PyHolder> m_task;
};

/**
 * @brief Similar to CppToPyAwaitable but will yield to other fibers when waiting for the coroutine to finish. Use this
 * once per loop at the main entry point for the asyncio loop
 *
 */
class PYBIND11_EXPORT BoostFibersMainPyAwaitable : public CppToPyAwaitable
{
  public:
    using CppToPyAwaitable::CppToPyAwaitable;

  protected:
    void tick() override
    {
        // Call the base class and then see if any fibers need processing by calling yield
        CppToPyAwaitable::tick();

        bool has_fibers = boost::fibers::has_ready_fibers();

        if (has_fibers)
        {
            // Yield to other fibers
            boost::this_fiber::yield();
        }
    }
};

class PYBIND11_EXPORT PyTaskToCppAwaitable
{
  public:
    PyTaskToCppAwaitable() = default;
    PyTaskToCppAwaitable(mrc::pymrc::PyObjectHolder&& task) : m_task(std::move(task)) {}

    bool await_ready() const noexcept
    {
        // Always suspend
        return false;
    }

    void await_suspend(std::coroutine_handle<> caller) noexcept
    {
        pybind11::gil_scoped_acquire gil;

        auto done_callback = pybind11::cpp_function([this, caller](pybind11::object future) {
            try
            {
                // Save the result value
                m_result = future.attr("result")();
            } catch (pybind11::error_already_set)
            {
                m_exception_ptr = std::current_exception();
            }

            pybind11::gil_scoped_release nogil;

            // Resume the coroutine
            caller.resume();
        });

        m_task.attr("add_done_callback")(done_callback);
    }

    mrc::pymrc::PyHolder await_resume()
    {
        if (m_exception_ptr)
        {
            std::rethrow_exception(m_exception_ptr);
        }

        return std::move(m_result);
    }

  private:
    mrc::pymrc::PyObjectHolder m_task;
    mrc::pymrc::PyHolder m_result;
    std::exception_ptr m_exception_ptr;
};

// ====== HELPER MACROS ======

#define MRC_PYBIND11_FAIL_ABSTRACT(cname, fnname)                                                                \
    pybind11::pybind11_fail(MRC_CONCAT_STR("Tried to call pure virtual function \"" << PYBIND11_STRINGIFY(cname) \
                                                                                    << "::" << fnname << "\""));

// ====== OVERRIDE PURE TEMPLATE ======
#define MRC_PYBIND11_OVERRIDE_PURE_TEMPLATE_NAME(ret_type, abstract_cname, cname, name, fn, ...)  \
    do                                                                                            \
    {                                                                                             \
        PYBIND11_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        if constexpr (std::is_same_v<cname, abstract_cname>)                                      \
        {                                                                                         \
            MRC_PYBIND11_FAIL_ABSTRACT(PYBIND11_TYPE(abstract_cname), name);                      \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            return cname::fn(__VA_ARGS__);                                                        \
        }                                                                                         \
    } while (false)

#define MRC_PYBIND11_OVERRIDE_PURE_TEMPLATE(ret_type, abstract_cname, cname, fn, ...) \
    MRC_PYBIND11_OVERRIDE_PURE_TEMPLATE_NAME(                                         \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(abstract_cname), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)
// ====== OVERRIDE PURE TEMPLATE ======

// ====== OVERRIDE COROUTINE IMPL ======
#define MRC_PYBIND11_OVERRIDE_CORO_IMPL(ret_type, cname, name, ...)                                          \
    do                                                                                                       \
    {                                                                                                        \
        DCHECK_EQ(PyGILState_Check(), 0) << "Should not have the GIL when resuming a C++ coroutine";         \
        pybind11::gil_scoped_acquire gil;                                                                    \
        pybind11::function override = pybind11::get_override(static_cast<const cname*>(this), name);         \
        if (override)                                                                                        \
        {                                                                                                    \
            auto o_coro         = override(__VA_ARGS__);                                                     \
            auto asyncio_module = pybind11::module::import("asyncio");                                       \
            /* Return type must be a coroutine to allow calling asyncio.create_task() */                     \
            if (!asyncio_module.attr("iscoroutine")(o_coro).cast<bool>())                                    \
            {                                                                                                \
                pybind11::pybind11_fail(MRC_CONCAT_STR("Return value from overriden async function "         \
                                                       << PYBIND11_STRINGIFY(cname) << "::" << name          \
                                                       << " did not return a coroutine. Returned: "          \
                                                       << pybind11::str(o_coro).cast<std::string>()));       \
            }                                                                                                \
            auto o_task = asyncio_module.attr("create_task")(o_coro);                                        \
            mrc::pymrc::PyHolder o_result;                                                                   \
            {                                                                                                \
                pybind11::gil_scoped_release nogil;                                                          \
                o_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(o_task));                    \
                DCHECK_EQ(PyGILState_Check(), 0) << "Should not have the GIL after returning from co_await"; \
            }                                                                                                \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value)                        \
            {                                                                                                \
                static pybind11::detail::override_caster_t<ret_type> caster;                                 \
                co_return pybind11::detail::cast_ref<ret_type>(std::move(o_result), caster);                 \
            }                                                                                                \
            co_return pybind11::detail::cast_safe<ret_type>(std::move(o_result));                            \
        }                                                                                                    \
    } while (false)
// ====== OVERRIDE COROUTINE IMPL======

// ====== OVERRIDE COROUTINE ======
#define MRC_PYBIND11_OVERRIDE_CORO_NAME(ret_type, cname, name, fn, ...)                                    \
    do                                                                                                     \
    {                                                                                                      \
        MRC_PYBIND11_OVERRIDE_CORO_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        return cname::fn(__VA_ARGS__);                                                                     \
    } while (false)

#define MRC_PYBIND11_OVERRIDE_CORO(ret_type, cname, fn, ...) \
    MRC_PYBIND11_OVERRIDE_CORO_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)
// ====== OVERRIDE COROUTINE ======

// ====== OVERRIDE COROUTINE PURE======
#define MRC_PYBIND11_OVERRIDE_CORO_PURE_NAME(ret_type, cname, name, fn, ...)                               \
    do                                                                                                     \
    {                                                                                                      \
        MRC_PYBIND11_OVERRIDE_CORO_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        MRC_PYBIND11_FAIL_ABSTRACT(PYBIND11_TYPE(cname), name);                                            \
    } while (false)

#define MRC_PYBIND11_OVERRIDE_CORO_PURE(ret_type, cname, fn, ...) \
    MRC_PYBIND11_OVERRIDE_CORO_PURE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)
// ====== OVERRIDE COROUTINE PURE======

// ====== OVERRIDE COROUTINE PURE TEMPLATE======
#define MRC_PYBIND11_OVERRIDE_CORO_PURE_TEMPLATE_NAME(ret_type, abstract_cname, cname, name, fn, ...)      \
    do                                                                                                     \
    {                                                                                                      \
        MRC_PYBIND11_OVERRIDE_CORO_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__); \
        if constexpr (std::is_same_v<cname, abstract_cname>)                                               \
        {                                                                                                  \
            MRC_PYBIND11_FAIL_ABSTRACT(PYBIND11_TYPE(abstract_cname), name);                               \
        }                                                                                                  \
        else                                                                                               \
        {                                                                                                  \
            co_return co_await cname::fn(__VA_ARGS__);                                                     \
        }                                                                                                  \
    } while (false)

#define MRC_PYBIND11_OVERRIDE_CORO_PURE_TEMPLATE(ret_type, abstract_cname, cname, fn, ...) \
    MRC_PYBIND11_OVERRIDE_CORO_PURE_TEMPLATE_NAME(                                         \
        PYBIND11_TYPE(ret_type), PYBIND11_TYPE(abstract_cname), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)
// ====== OVERRIDE COROUTINE PURE TEMPLATE======

}  // namespace mrc::pycoro

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

/**
 * @brief Provides a type caster for converting a C++ coroutine to a python awaitable. Include this file in any pybind11
 * module to automatically convert the types. Allows for converting arguments and return values.
 *
 * @tparam ReturnT The return type of the coroutine
 */
template <typename ReturnT>
struct type_caster<mrc::coroutines::Task<ReturnT>>
{
  public:
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(mrc::coroutines::Task<ReturnT>, _("typing.Awaitable[") + make_caster<ReturnT>::name + _("]"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool convert)
    {
        if (!src || src.is_none())
        {
            return false;
        }

        if (!PyCoro_CheckExact(src.ptr()))
        {
            return false;
        }

        auto cpp_coro = [](mrc::pymrc::PyHolder py_task) -> mrc::coroutines::Task<ReturnT> {
            DCHECK_EQ(PyGILState_Check(), 0) << "Should not have the GIL when resuming a C++ coroutine";

            // Always assume we are resuming without the GIL
            pybind11::gil_scoped_acquire gil;

            auto asyncio_task = pybind11::module_::import("asyncio").attr("create_task")(py_task);

            mrc::pymrc::PyHolder py_result;
            {
                // Release the GIL before awaiting
                pybind11::gil_scoped_release nogil;

                py_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(asyncio_task));
            }

            // Now cast back to the C++ type
            if (pybind11::detail::cast_is_temporary_value_reference<ReturnT>::value)
            {
                static pybind11::detail::override_caster_t<ReturnT> caster;
                co_return pybind11::detail::cast_ref<ReturnT>(std::move(py_result), caster);
            }
            co_return pybind11::detail::cast_safe<ReturnT>(std::move(py_result));
        };

        value = cpp_coro(pybind11::reinterpret_borrow<pybind11::object>(std::move(src)));

        return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(mrc::coroutines::Task<ReturnT> src, return_value_policy policy, handle parent)
    {
        // Wrap the object in a CppToPyAwaitable
        std::shared_ptr<mrc::pycoro::CppToPyAwaitable> awaitable =
            std::make_shared<mrc::pycoro::CppToPyAwaitable>(std::move(src));

        // Convert the object to a python object
        auto py_awaitable = pybind11::cast(std::move(awaitable));

        return py_awaitable.release();
    }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
