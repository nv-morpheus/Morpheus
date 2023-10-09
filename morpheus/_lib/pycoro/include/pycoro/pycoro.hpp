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

}  // namespace mrc::pycoro
