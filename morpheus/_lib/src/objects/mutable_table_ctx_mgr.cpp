#include "morpheus/objects/mutable_table_ctx_mgr.hpp"

#include <pybind11/gil.h>

namespace morpheus {

namespace py = pybind11;

/********** MutableTableCtxMgr **********/
MutableTableCtxMgr::MutableTableCtxMgr(MessageMeta& meta_msg) :
  m_meta_msg{meta_msg},
  m_table{nullptr},
  m_py_table{nullptr} {};

py::object MutableTableCtxMgr::enter()
{
    // Release the GIL
    py::gil_scoped_release no_gil;
    m_table    = std::make_unique<MutableTableInfo>(std::move(m_meta_msg.get_mutable_info()));
    m_py_table = std::make_unique<py::object>(std::move(m_table->checkout_obj()));
    return *m_py_table;
}

void MutableTableCtxMgr::exit(const py::object& type, const py::object& value, const py::object& traceback)
{
    m_table->return_obj(std::move(*m_py_table.release()));
    m_table.reset(nullptr);
}

void MutableTableCtxMgr::throw_usage_error(pybind11::args args, const pybind11::kwargs& kwargs)
{
    std::ostringstream err_msg;
    err_msg << "Error attempting to use mutable_dataframe outside of context manager. Intended usage :\n";
    err_msg << "with message_meta.mutable_dataframe() as df:\n";
    err_msg << "    df['col'] = 5";
    throw py::attribute_error(err_msg.str());
}

}  // namespace morpheus
