#include <pybind11/cast.h>  // for object_api::operator()
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_

#include <mrc/segment/builder.hpp>

#include <morpheus/stages/doca_source.hpp>

namespace morpheus {
  
namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(doca, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<DocaSourceStage>,
              mrc::segment::ObjectProperties,
              std::shared_ptr<mrc::segment::Object<DocaSourceStage>>>(m, "DocaSourceStage", py::multiple_inheritance())
        .def(py::init<>(&DocaSourceStageInterfaceProxy::init),
            py::arg("builder"),
            py::arg("name"),
            py::arg("nic_pci_address"),
            py::arg("gpu_pci_address"),
            py::arg("source_ip_filter") = "");
}

}
