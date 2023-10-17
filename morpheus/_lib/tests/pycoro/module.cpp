#include "pycoro/pycoro.hpp"

#include <mrc/coroutines/task.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/types.hpp>

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

PYBIND11_MODULE(pycoro, _module)
{
    _module.def("call_fib_async", &call_fib_async);
}

}  // namespace morpheus::tests::pycoro
