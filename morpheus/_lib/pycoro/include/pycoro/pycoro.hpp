#pragma once

#include "morpheus/export.h"

#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pymrc/types.hpp>

#include <memory>

namespace mrc::pycoro {

class MORPHEUS_EXPORT StopIteration : public pybind11::stop_iteration
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

class MORPHEUS_EXPORT CppToPyAwaitable : public std::enable_shared_from_this<CppToPyAwaitable>
{
  public:
    CppToPyAwaitable() = default;

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

        if (!m_has_resumed)
        {
            m_has_resumed = true;

            m_task.resume();
        }

        if (m_task.is_ready())
        {
            // Grab the gil before moving and throwing
            pybind11::gil_scoped_acquire gil;

            // job done -> throw
            auto exception = StopIteration(std::move(m_task.promise().result()));

            // Destroy the task now that we have the value
            m_task.destroy();

            throw exception;
        }
    }

  private:
    bool m_has_resumed{false};
    mrc::coroutines::Task<mrc::pymrc::PyHolder> m_task;
};

class MORPHEUS_EXPORT PyTaskToCppAwaitable
{
    struct Awaiter
    {
        Awaiter(const PyTaskToCppAwaitable& parent) noexcept : m_parent(parent) {}

        bool await_ready() const noexcept
        {
            // pybind11::gil_scoped_acquire gil;

            // return m_parent.m_task.attr("done")().cast<bool>();

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

            m_parent.m_task.attr("add_done_callback")(done_callback);
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
        const PyTaskToCppAwaitable& m_parent;
        mrc::pymrc::PyHolder m_result;
        std::exception_ptr m_exception_ptr;
    };

  public:
    PyTaskToCppAwaitable() = default;
    PyTaskToCppAwaitable(mrc::pymrc::PyObjectHolder&& task) : m_task(std::move(task)) {}

    Awaiter operator co_await() const noexcept
    {
        return Awaiter{*this};
    }

  private:
    mrc::pymrc::PyObjectHolder m_task;

    friend struct Awaiter;
};

namespace mrc::pycoro {

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
#define MRC_PYBIND11_OVERRIDE_CORO_IMPL(ret_type, cname, name, ...)                                    \
    do                                                                                                 \
    {                                                                                                  \
        pybind11::gil_scoped_acquire gil;                                                              \
        pybind11::function override = pybind11::get_override(static_cast<const cname*>(this), name);   \
        if (override)                                                                                  \
        {                                                                                              \
            auto o_coro         = override(__VA_ARGS__);                                               \
            auto asyncio_module = pybind11::module::import("asyncio");                                 \
            /* Return type must be a coroutine to allow calling asyncio.create_task() */               \
            if (!asyncio_module.attr("iscoroutine")(o_coro).cast<bool>())                              \
            {                                                                                          \
                pybind11::pybind11_fail(MRC_CONCAT_STR("Return value from overriden async function "   \
                                                       << PYBIND11_STRINGIFY(cname) << "::" << name    \
                                                       << " did not return a coroutine. Returned: "    \
                                                       << pybind11::str(o_coro).cast<std::string>())); \
            }                                                                                          \
            auto o_task = asyncio_module.attr("create_task")(o_coro);                                  \
            mrc::pymrc::PyHolder o_result;                                                             \
            {                                                                                          \
                pybind11::gil_scoped_release nogil;                                                    \
                o_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(o_task));              \
            }                                                                                          \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value)                  \
            {                                                                                          \
                static pybind11::detail::override_caster_t<ret_type> caster;                           \
                co_return pybind11::detail::cast_ref<ret_type>(std::move(o_result), caster);           \
            }                                                                                          \
            co_return pybind11::detail::cast_safe<ret_type>(std::move(o_result));                      \
        }                                                                                              \
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

}  // namespace mrc::pycoro
