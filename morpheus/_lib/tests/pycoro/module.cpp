
#include <pybind11/pybind11.h>

namespace morpheus::tests::pycoro {

void hey() {}

PYBIND11_MODULE(pycoro, _module)
{
    _module.def("hey", &hey);
}

}  // namespace morpheus::tests::pycoro
