/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./include/py_llm_engine.hpp"
#include "./include/py_llm_node.hpp"
#include "./include/py_llm_node_base.hpp"
#include "./include/py_llm_task_handler.hpp"
#include "pycoro/pycoro.hpp"

#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_node_base.hpp"
#include "morpheus/llm/llm_node_runner.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/version.hpp"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>
#include <mrc/segment/object.hpp>
#include <mrc/types.hpp>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/attr.h>  // for multiple_inheritance
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for arg, init, class_, module_, str_attr_accessor, PYBIND11_MODULE, pybind11
#include <pybind11/pytypes.h>   // for dict, sequence
#include <pybind11/stl.h>
#include <pymrc/types.hpp>
#include <pymrc/utilities/acquire_gil.hpp>
#include <pymrc/utilities/function_wrappers.hpp>
#include <pymrc/utilities/object_wrappers.hpp>
#include <pymrc/utils.hpp>  // for pymrc::import
#include <rxcpp/rx.hpp>

#include <chrono>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// namespace mrc::pycoro {
// namespace py = pybind11;

// class StopIteration : public py::stop_iteration
// {
//   public:
//     StopIteration(py::object&& result) : stop_iteration("--"), m_result(std::move(result)){};

//     void set_error() const override
//     {
//         PyErr_SetObject(PyExc_StopIteration, this->m_result.ptr());
//     }

//   private:
//     py::object m_result;
// };

// class CppToPyAwaitable : public std::enable_shared_from_this<CppToPyAwaitable>
// {
//   public:
//     CppToPyAwaitable() = default;

//     CppToPyAwaitable(mrc::coroutines::Task<mrc::pymrc::PyHolder>&& task) : m_task(std::move(task)) {}

//     std::shared_ptr<CppToPyAwaitable> iter()
//     {
//         return this->shared_from_this();
//     }

//     std::shared_ptr<CppToPyAwaitable> await()
//     {
//         return this->shared_from_this();
//     }

//     void next()
//     {
//         // Need to release the GIL before waiting
//         py::gil_scoped_release nogil;

//         if (!m_has_resumed)
//         {
//             m_has_resumed = true;

//             m_task.resume();
//         }

//         if (m_task.is_ready())
//         {
//             // Grab the gil before moving and throwing
//             py::gil_scoped_acquire gil;

//             // job done -> throw
//             auto exception = StopIteration(std::move(m_task.promise().result()));

//             // Destroy the task now that we have the value
//             m_task.destroy();

//             throw exception;
//         }
//     }

//   private:
//     bool m_has_resumed{false};
//     mrc::coroutines::Task<mrc::pymrc::PyHolder> m_task;
// };

// class PyTaskToCppAwaitable
// {
//     struct Awaiter
//     {
//         Awaiter(const PyTaskToCppAwaitable& parent) noexcept : m_parent(parent) {}

//         bool await_ready() const noexcept
//         {
//             // pybind11::gil_scoped_acquire gil;

//             // return m_parent.m_task.attr("done")().cast<bool>();

//             // Always suspend
//             return false;
//         }

//         void await_suspend(std::coroutine_handle<> caller) noexcept
//         {
//             pybind11::gil_scoped_acquire gil;

//             auto done_callback = py::cpp_function([this, caller](pybind11::object future) {
//                 try
//                 {
//                     // Save the result value
//                     m_result = future.attr("result")();
//                 } catch (py::error_already_set)
//                 {
//                     m_exception_ptr = std::current_exception();
//                 }

//                 pybind11::gil_scoped_release nogil;

//                 // Resume the coroutine
//                 caller.resume();
//             });

//             m_parent.m_task.attr("add_done_callback")(done_callback);
//         }

//         mrc::pymrc::PyHolder await_resume()
//         {
//             if (m_exception_ptr)
//             {
//                 std::rethrow_exception(m_exception_ptr);
//             }

//             return std::move(m_result);
//         }

//       private:
//         const PyTaskToCppAwaitable& m_parent;
//         mrc::pymrc::PyHolder m_result;
//         std::exception_ptr m_exception_ptr;
//     };

//   public:
//     PyTaskToCppAwaitable() = default;
//     PyTaskToCppAwaitable(mrc::pymrc::PyObjectHolder&& task) : m_task(std::move(task)) {}

//     Awaiter operator co_await() const noexcept
//     {
//         return Awaiter{*this};
//     }

//   private:
//     mrc::pymrc::PyObjectHolder m_task;

//     friend struct Awaiter;
// };

// }  // namespace mrc::pycoro

namespace morpheus::llm {
namespace py = pybind11;

// std::function<void()> create_gil_initializer()
// {
//     bool has_pydevd_trace = false;

//     // We check if there is a debugger by looking at sys.gettrace() and seeing if the function contains 'pydevd'
//     // somewhere in the module name. Its important to get this right because calling `debugpy.debug_this_thread()`
//     // will fail if there is no debugger and can dramatically alter performanc
//     auto sys = pybind11::module_::import("sys");

//     auto trace_func = sys.attr("gettrace")();

//     py::print("Trace func: ", trace_func);

//     if (!trace_func.is_none())
//     {
//         // Convert it to a string to quickly get its module and name
//         auto trace_func_str = pybind11::str(trace_func);

//         if (!trace_func_str.attr("find")("pydevd").equal(pybind11::int_(-1)))
//         {
//             VLOG(10) << "Found pydevd trace function. Will attempt to enable debugging for MRC threads.";
//             has_pydevd_trace = true;
//         }
//     }
//     else
//     {
//         VLOG(10) << "Not setting up debugging. No trace function found.";
//     }

//     return [has_pydevd_trace] {
//         pybind11::gil_scoped_acquire gil;

//         // // Increment the ref once to prevent creating and destroying the thread state constantly
//         // gil.inc_ref();

//         try
//         {
//             // Try to load debugpy only if we found a trace function
//             if (has_pydevd_trace)
//             {
//                 auto debugpy = pybind11::module_::import("debugpy");

//                 auto debug_this_thread = debugpy.attr("debug_this_thread");

//                 debug_this_thread();

//                 VLOG(10) << "Debugging enabled from mrc threads";
//             }
//         } catch (const pybind11::error_already_set& err)
//         {
//             if (err.matches(PyExc_ImportError))
//             {
//                 VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Could import error on debugpy";
//                 // Fail silently
//             }
//             else
//             {
//                 VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Unknown error: " << err.what();
//                 // Rethrow everything else
//                 throw;
//             }
//         }
//     };
// }

// class PyLLMService : public LLMService
// {
//   public:
//     LLMGenerateResult generate(LLMGeneratePrompt prompt) const override
//     {
//         using return_t = LLMGenerateResult;

//         pybind11 ::gil_scoped_acquire gil;

//         pybind11 ::function override = pybind11 ::get_override(static_cast<const llm ::LLMService*>(this),
//         "generate");

//         if (!override)
//         {
//             // Problem
//             pybind11 ::pybind11_fail(
//                 "Tried to call pure virtual function \""
//                 "LLMService"
//                 "::"
//                 "generate"
//                 "\"");
//         }

//         auto override_result = override(prompt);

//         // Now determine if the override result is a coroutine or not
//         if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
//         {
//             py::print("Returned a coroutine");

//             // Need to schedule the result to run on the loop
//             auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, m_loop);

//             // We are a dask future. Quickly check if its done, then release
//             while (!future.attr("done")().cast<bool>())
//             {
//                 // Release the GIL and wait for it to be done
//                 py::gil_scoped_release nogil;

//                 boost::this_fiber::yield();
//             }

//             // Completed, move into the returned object
//             override_result = future.attr("result")();
//         }
//         else
//         {
//             py::print("Did not return a coroutine");
//         }

//         // Now cast back to the C++ type
//         if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
//         {
//             static pybind11 ::detail ::override_caster_t<return_t> caster;
//             return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
//         }
//         return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
//     }

//   private:
//     void set_loop(py::object loop)
//     {
//         m_loop = std::move(loop);
//     }

//     mrc::pymrc::PyHolder m_loop;

//     friend class PyLLMEngine;
// };

// class PyLLMTaskHandler : public LLMTaskHandler
// {
//   public:
//     using LLMTaskHandler::LLMTaskHandler;

//     std::vector<std::string> get_input_names() const override
//     {
//         PYBIND11_OVERRIDE_PURE(std::vector<std::string>, LLMTaskHandler, get_input_names);
//     }

//     Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context) override
//     {
//         using return_t = LLMTaskHandler::return_t;

//         pybind11 ::gil_scoped_acquire gil;

//         pybind11 ::function override =
//             pybind11 ::get_override(static_cast<const llm ::LLMTaskHandler*>(this), "try_handle");

//         if (!override)
//         {
//             // Problem
//             pybind11 ::pybind11_fail(
//                 "Tried to call pure virtual function \""
//                 "LLMTaskHandler"
//                 "::"
//                 "try_handle"
//                 "\"");
//         }

//         auto override_coro = override(context);

//         // Now determine if the override result is a coroutine or not
//         if (!py::module::import("asyncio").attr("iscoroutine")(override_coro).cast<bool>())
//         {
//             throw std::runtime_error("Must return a coroutine");
//         }

//         auto override_task = py::module::import("asyncio").attr("create_task")(override_coro);

//         mrc::pymrc::PyHolder override_result;
//         {
//             // Release the GIL before awaiting
//             pybind11::gil_scoped_release nogil;

//             override_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(override_task));
//         }

//         // Now cast back to the C++ type
//         if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
//         {
//             static pybind11 ::detail ::override_caster_t<return_t> caster;
//             co_return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
//         }
//         co_return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
//     }

//   private:
//     void set_loop(py::object loop)
//     {
//         m_loop = std::move(loop);
//     }

//     mrc::pymrc::PyHolder m_loop;

//     friend class PyLLMEngine;
// };

// template <class BaseT = LLMNodeBase>
// class PyLLMNodeBase : public BaseT
// {
//   public:
//     using BaseT::BaseT;

//     // std::shared_ptr<LLMNodeRunner> add_node(std::string name,
//     //                                              std::vector<std::string> input_names,
//     //                                              std::shared_ptr<LLMNodeBase> node) override
//     // {
//     //     // // Try to cast the object to a python object to ensure that we keep it alive
//     //     // auto py_node = std::dynamic_pointer_cast<PyLLMNodeBase>(node);

//     //     // if (py_node)
//     //     // {
//     //     // Store the python object to keep it alive
//     //     m_py_nodes[node] = py::cast(node);
//     //     // }

//     //     // Call the base class implementation
//     //     return LLMNode::add_node(name, input_names, node);
//     // }

//     std::vector<std::string> get_input_names() const override
//     {
//         // Call the python overridden function

//         pybind11 ::gil_scoped_acquire gil;
//         pybind11 ::function override = pybind11 ::get_override(static_cast<const BaseT*>(this), "get_input_names");

//         if (override)
//         {
//             auto o = override();
//             if (pybind11 ::detail ::cast_is_temporary_value_reference<std ::vector<std ::string>>::value)
//             {
//                 static pybind11 ::detail ::override_caster_t<std ::vector<std ::string>> caster;
//                 return pybind11 ::detail ::cast_ref<std ::vector<std ::string>>(std ::move(o), caster);
//             }
//             return pybind11 ::detail ::cast_safe<std ::vector<std ::string>>(std ::move(o));
//         }

//         if constexpr (std::is_same_v<BaseT, LLMNodeBase>)
//         {
//             // Cant call the base class implementation on abstract class
//             pybind11::pybind11_fail(
//                 "Tried to call pure virtual function \""
//                 "LLMNodeBase"
//                 "::"
//                 "get_input_names"
//                 "\"");
//         }
//         else
//         {
//             return BaseT::get_input_names();
//         }
//     }

//     Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
//     {
//         using return_t = std::shared_ptr<LLMContext>;

//         pybind11::gil_scoped_acquire gil;

//         pybind11::function override = pybind11::get_override(static_cast<const BaseT*>(this), "execute");

//         if (override)
//         {
//             auto override_coro = override(context);

//             // Now determine if the override result is a coroutine or not
//             if (!py::module::import("asyncio").attr("iscoroutine")(override_coro).cast<bool>())
//             {
//                 throw std::runtime_error("Must return a coroutine");
//             }

//             auto override_task = py::module::import("asyncio").attr("create_task")(override_coro);

//             mrc::pymrc::PyHolder override_result;
//             {
//                 // Release the GIL before awaiting
//                 pybind11::gil_scoped_release nogil;

//                 override_result = co_await mrc::pycoro::PyTaskToCppAwaitable(std::move(override_task));
//             }

//             // Now cast back to the C++ type
//             if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
//             {
//                 static pybind11 ::detail ::override_caster_t<return_t> caster;
//                 co_return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
//             }
//             co_return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
//         }

//         if constexpr (std::is_same_v<BaseT, LLMNodeBase>)
//         {
//             // Cant call the base class implementation on abstract class
//             pybind11::pybind11_fail(
//                 "Tried to call pure virtual function \""
//                 "LLMNodeBase"
//                 "::"
//                 "execute"
//                 "\"");
//         }
//         else
//         {
//             co_return co_await BaseT::execute(context);
//         }
//     }

//   private:
//     std::map<std::shared_ptr<LLMNodeBase>, py::object> m_py_nodes;
// };

// template <class BaseT = LLMNode>
// class PyLLMNode : public PyLLMNodeBase<BaseT>
// {
//   public:
//     using PyLLMNodeBase<BaseT>::PyLLMNodeBase;

//     std::shared_ptr<LLMNodeRunner> add_node(std::string name,
//                                             input_map_t inputs,
//                                             std::shared_ptr<LLMNodeBase> node,
//                                             bool is_output = false) override
//     {
//         // // Try to cast the object to a python object to ensure that we keep it alive
//         // auto py_node = std::dynamic_pointer_cast<PyLLMNodeBase>(node);

//         // if (py_node)
//         // {
//         // Store the python object to keep it alive
//         m_py_nodes[node] = py::cast(node);
//         // }

//         // Call the base class implementation
//         return LLMNode::add_node(std::move(name), std::move(inputs), std::move(node), is_output);
//     }

//     // void execute(std::shared_ptr<LLMContext> context) override
//     // {
//     //     pybind11 ::gil_scoped_acquire gil;

//     //     pybind11 ::function override = pybind11 ::get_override(static_cast<const BaseT*>(this), "execute");

//     //     if (!override)
//     //     {
//     //         // Problem
//     //         pybind11 ::pybind11_fail(
//     //             "Tried to call pure virtual function \""
//     //             "LLMNodeBase"
//     //             "::"
//     //             "execute"
//     //             "\"");
//     //     }

//     //     auto override_result = override(context);

//     //     // Now determine if the override result is a coroutine or not
//     //     if (py::module::import("asyncio").attr("iscoroutine")(override_result).cast<bool>())
//     //     {
//     //         py::print("Returned a coroutine");

//     //         auto loop = py::module::import("asyncio").attr("get_running_loop")();

//     //         // Need to schedule the result to run on the loop
//     //         auto future = py::module::import("asyncio").attr("run_coroutine_threadsafe")(override_result, loop);

//     //         // We are a dask future. Quickly check if its done, then release
//     //         while (!future.attr("done")().cast<bool>())
//     //         {
//     //             // Release the GIL and wait for it to be done
//     //             py::gil_scoped_release nogil;

//     //             boost::this_fiber::yield();
//     //         }

//     //         // // Completed, move into the returned object
//     //         // override_result = future.attr("result")();
//     //     }
//     //     else
//     //     {
//     //         py::print("Did not return a coroutine");
//     //     }

//     //     // // Now cast back to the C++ type
//     //     // if (pybind11 ::detail ::cast_is_temporary_value_reference<return_t>::value)
//     //     // {
//     //     //     static pybind11 ::detail ::override_caster_t<return_t> caster;
//     //     //     return pybind11 ::detail ::cast_ref<return_t>(std ::move(override_result), caster);
//     //     // }
//     //     // return pybind11 ::detail ::cast_safe<return_t>(std ::move(override_result));
//     // }

//   private:
//     std::map<std::shared_ptr<LLMNodeBase>, py::object> m_py_nodes;
// };

// class PyLLMEngine : public PyLLMNode<LLMEngine>
// {
//   public:
//     PyLLMEngine() : PyLLMNode<LLMEngine>()
//     {
//         // std::promise<void> loop_ready;

//         // auto future = loop_ready.get_future();

//         // auto setup_debugging = create_gil_initializer();

//         // m_thread = std::thread(
//         //     [this](std::promise<void> loop_ready, std::function<void()> setup_debugging) {
//         //         // Acquire the GIL (and also initialize the ThreadState)
//         //         py::gil_scoped_acquire acquire;

//         //         // Initialize the debugger
//         //         setup_debugging();

//         //         py::print("Creating loop");

//         //         // Gets (or more likely, creates) an event loop and runs it forever until stop is called
//         //         m_loop = py::module::import("asyncio").attr("new_event_loop")();

//         //         py::print("Setting loop current");

//         //         // Set the event loop as the current event loop
//         //         py::module::import("asyncio").attr("set_event_loop")(m_loop);

//         //         py::print("Signaling promise");

//         //         // Signal we are ready
//         //         loop_ready.set_value();

//         //         py::print("Running forever");

//         //         m_loop.attr("run_forever")();
//         //     },
//         //     std::move(loop_ready),
//         //     std::move(setup_debugging));

//         // py::print("Waiting for startup");
//         // {
//         //     // Free the GIL otherwise we deadlock
//         //     py::gil_scoped_release nogil;

//         //     future.get();
//         // }

//         // // Finally, try and see if our LLM Service is a python object and keep it alive
//         // auto py_llm_service = std::dynamic_pointer_cast<PyLLMService>(llm_service);

//         // if (py_llm_service)
//         // {
//         //     // Store the python object to keep it alive
//         //     m_py_llm_service = py::cast(llm_service);

//         //     // Also, set the loop on the service
//         //     py_llm_service->set_loop(m_loop);
//         // }

//         // py::print("Engine started");
//     }

//     ~PyLLMEngine() override
//     {
//         // Acquire the GIL on this thread and call stop on the event loop
//         py::gil_scoped_acquire acquire;

//         m_loop.attr("stop")();

//         // Finally, join on the thread
//         m_thread.join();
//     }

//     void add_task_handler(input_map_t inputs, std::shared_ptr<LLMTaskHandler> task_handler) override
//     {
//         // Try to cast the object to a python object to ensure that we keep it alive
//         auto py_task_handler = std::dynamic_pointer_cast<PyLLMTaskHandler>(task_handler);

//         if (py_task_handler)
//         {
//             // Store the python object to keep it alive
//             m_py_task_handler[task_handler] = py::cast(task_handler);
//         }

//         // Call the base class implementation
//         LLMEngine::add_task_handler(std::move(inputs), std::move(task_handler));
//     }

//     const py::object& get_loop() const
//     {
//         return m_loop;
//     }

//     // std::vector<std::shared_ptr<ControlMessage>> run(std::shared_ptr<ControlMessage> input_message) override
//     // {
//     //     std::vector<std::shared_ptr<ControlMessage>> output_messages;

//     //     return output_messages;
//     // }

//   private:
//     std::thread m_thread;
//     py::object m_loop;

//     // Keep the python objects alive by saving references in this object
//     py::object m_py_llm_service;
//     std::map<std::shared_ptr<LLMTaskHandler>, py::object> m_py_task_handler;
// };

PYBIND11_MODULE(llm, _module)
{
    _module.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        )pbdoc";

    // Load the cudf helpers
    CudfHelper::load();

    // Import the pycoro module
    py::module_ pycoro = py::module_::import("morpheus._lib.pycoro");

    py::class_<InputMap>(_module, "InputMap")
        .def(py::init<>())
        .def_readwrite("input_name", &InputMap::input_name)
        .def_readwrite("node_name", &InputMap::node_name);

    py::class_<LLMTask>(_module, "LLMTask")
        .def(py::init<>())
        .def(py::init([](std::string task_type, py::dict task_dict) {
            return LLMTask(std::move(task_type), mrc::pymrc::cast_from_pyobject(task_dict));
        }))
        .def_readonly("task_type", &LLMTask::task_type)
        .def("__getitem__",
             [](const LLMTask& self, const std::string& key) {
                 try
                 {
                     return mrc::pymrc::cast_from_json(self.get(key));
                 } catch (const std::out_of_range&)
                 {
                     throw py::key_error("key '" + key + "' does not exist");
                 }
             })
        .def("__setitem__",
             [](LLMTask& self, const std::string& key, py::object value) {
                 try
                 {
                     // Convert to C++ nholman object

                     return self.set(key, mrc::pymrc::cast_from_pyobject(std::move(value)));
                 } catch (const std::out_of_range&)
                 {
                     throw py::key_error("key '" + key + "' does not exist");
                 }
             })
        .def("__len__", &LLMTask::size)
        .def("get", [](const LLMTask& self, const std::string& key, py::object default_value) {
            try
            {
                return mrc::pymrc::cast_from_json(self.get(key));
            } catch (const nlohmann::detail::out_of_range&)
            {
                return default_value;
            }
        });
    // .def(
    //     "__iter__",
    //     [](const StringMap& map) { return py::make_key_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "items",
    //     [](const StringMap& map) { return py::make_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "values",
    //     [](const StringMap& map) { return py::make_value_iterator(map.begin(), map.end()); },
    //     py::keep_alive<0, 1>());

    py::class_<LLMContext, std::shared_ptr<LLMContext>>(_module, "LLMContext")
        .def_property_readonly("name", [](LLMContext& self) { return self.name(); })
        .def_property_readonly("full_name", [](LLMContext& self) { return self.full_name(); })
        .def_property_readonly("all_outputs",
                               [](LLMContext& self) {
                                   return mrc::pymrc::cast_from_json(self.all_outputs());
                               })  // Remove all_outputs before release!
        .def_property_readonly("input_map",
                               [](LLMContext& self) {
                                   //
                                   return self.input_map();
                               })
        .def_property_readonly("parent",
                               [](LLMContext& self) {
                                   //
                                   return self.parent();
                               })
        .def("task", [](LLMContext& self) { return self.task(); })
        .def("message", [](LLMContext& self) { return self.message(); })
        .def("get_input",
             [](LLMContext& self) {
                 // Convert the return value
                 return mrc::pymrc::cast_from_json(self.get_input());
             })
        .def("get_input",
             [](LLMContext& self, std::string key) {
                 // Convert the return value
                 return mrc::pymrc::cast_from_json(self.get_input(key));
             })
        .def("get_inputs",
             [](LLMContext& self) {
                 // Convert the return value
                 return mrc::pymrc::cast_from_json(self.get_inputs()).cast<py::dict>();
             })
        .def("set_output", [](LLMContext& self, py::object value) {
            // Convert and pass to the base
            self.set_output(mrc::pymrc::cast_from_pyobject(value));
        });

    py::class_<LLMNodeBase, PyLLMNodeBase<>, std::shared_ptr<LLMNodeBase>>(_module, "LLMNodeBase")
        .def(py::init_alias<>())
        .def("execute", [](std::shared_ptr<LLMNodeBase> self, std::shared_ptr<LLMContext> context) {
            auto convert = [self](std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
                auto result = co_await self->execute(context);

                // Convert the output to a python object
                co_return py::cast(result);
            };

            return std::make_shared<mrc::pycoro::CppToPyAwaitable>(convert(context));
        });

    py::class_<LLMNodeRunner, std::shared_ptr<LLMNodeRunner>>(_module, "LLMNodeRunner")
        .def_property_readonly("name", &LLMNodeRunner::name)
        .def_property_readonly("inputs", &LLMNodeRunner::inputs);
    // .def("execute", [](std::shared_ptr<LLMNodeRunner> self, std::shared_ptr<LLMContext> context) {
    //     auto convert = [self](std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
    //         auto result = co_await self->execute(context);

    //         // Convert the output to a python object
    //         co_return py::cast(result);
    //     };

    //     return std::make_shared<CoroAwaitable>(std::move(convert));
    // });

    py::class_<LLMNode, LLMNodeBase, PyLLMNode<>, std::shared_ptr<LLMNode>>(_module, "LLMNode")
        .def(py::init_alias<>())
        .def(
            "add_node",
            [](LLMNode& self, std::string name, std::shared_ptr<LLMNodeBase> node, bool is_output) {
                input_map_t converted_inputs;

                // Populate the inputs from the node input_names
                for (const auto& single_input : node->get_input_names())
                {
                    converted_inputs.push_back({.input_name = single_input});
                }

                return self.add_node(std::move(name), std::move(converted_inputs), std::move(node), is_output);
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("node"),
            py::arg("is_output") = false)
        .def(
            "add_node",
            [](LLMNode& self,
               std::string name,
               std::vector<std::variant<std::string, std::pair<std::string, std::string>>> inputs,
               std::shared_ptr<LLMNodeBase> node,
               bool is_output) {
                input_map_t converted_inputs;

                for (const auto& single_input : inputs)
                {
                    if (std::holds_alternative<std::string>(single_input))
                    {
                        converted_inputs.push_back({.input_name = std::get<std::string>(single_input)});
                    }
                    else
                    {
                        auto pair = std::get<std::pair<std::string, std::string>>(single_input);

                        converted_inputs.push_back({.input_name = pair.first, .node_name = pair.second});
                    }
                }

                return self.add_node(std::move(name), std::move(converted_inputs), std::move(node), is_output);
            },
            py::arg("name"),
            py::kw_only(),
            py::arg("inputs"),
            py::arg("node"),
            py::arg("is_output") = false);
    // .def("execute", [](std::shared_ptr<LLMNode> self, std::shared_ptr<LLMContext> context) {
    //     auto convert = [self](std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
    //         auto result = co_await self->execute(context);

    //         // Convert the output to a python object
    //         co_return py::cast(result);
    //     };

    //     return std::make_shared<CoroAwaitable>(std::move(convert));
    // });

    // auto LLMService =
    //     py::class_<LLMService, PyLLMService, std::shared_ptr<LLMService>>(_module, "LLMService");

    // auto LLMPromptGenerator =
    //     py::class_<LLMPromptGenerator, PyLLMPromptGenerator, std::shared_ptr<LLMPromptGenerator>>(
    //         _module, "LLMPromptGenerator");

    py::class_<LLMTaskHandler, PyLLMTaskHandler, std::shared_ptr<LLMTaskHandler>>(_module, "LLMTaskHandler")
        .def(py::init<>())
        .def(
            "try_handle",
            [](std::shared_ptr<LLMTaskHandler> self, std::shared_ptr<LLMContext> context) {
                auto convert = [](std::shared_ptr<LLMTaskHandler> self,
                                  std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
                    VLOG(10) << "Running LLMEngine::run";

                    auto result = co_await self->try_handle(context);

                    py::gil_scoped_acquire gil;

                    // Convert the output to a python object
                    co_return py::cast(result);
                };

                return std::make_shared<mrc::pycoro::CppToPyAwaitable>(convert(self, context));
            },
            py::arg("context"));

    py::class_<LLMEngine, LLMNode, PyLLMEngine, std::shared_ptr<LLMEngine>>(_module, "LLMEngine")
        .def(py::init_alias<>())
        // .def(py::init([](std::shared_ptr<LLMService> llm_service) {
        //     return std::make_shared<PyLLMEngine>(std::move(llm_service));
        // }))
        .def(
            "add_task_handler",
            [](LLMEngine& self,
               std::vector<std::variant<std::string, std::pair<std::string, std::string>>> inputs,
               std::shared_ptr<LLMTaskHandler> handler) {
                input_map_t converted_inputs;

                for (const auto& single_input : inputs)
                {
                    if (std::holds_alternative<std::string>(single_input))
                    {
                        converted_inputs.push_back({.input_name = std::get<std::string>(single_input)});
                    }
                    else
                    {
                        auto pair = std::get<std::pair<std::string, std::string>>(single_input);

                        converted_inputs.push_back({.input_name = pair.first, .node_name = pair.second});
                    }
                }

                return self.add_task_handler(std::move(converted_inputs), std::move(handler));
            },
            py::arg("inputs"),
            py::arg("handler"))
        // .def("add_node", &LLMEngine::add_node, py::arg("name"), py::arg("input_names"), py::arg("node"))
        .def(
            "run",
            [](std::shared_ptr<LLMEngine> self, std::shared_ptr<ControlMessage> message) {
                auto convert = [](std::shared_ptr<LLMEngine> self,
                                  std::shared_ptr<ControlMessage> message) -> Task<mrc::pymrc::PyHolder> {
                    VLOG(10) << "Running LLMEngine::run";

                    auto result = co_await self->run(message);

                    py::gil_scoped_acquire gil;

                    // Convert the output to a python object
                    co_return py::cast(message);
                };

                return std::make_shared<mrc::pycoro::CppToPyAwaitable>(convert(self, message));
            },
            py::arg("input_message"));

    // .def("execute", [](std::shared_ptr<LLMEngine> self, std::shared_ptr<LLMContext> context) {
    //     auto convert = [self](std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
    //         auto result = co_await self->execute(context);

    //         // Convert the output to a python object
    //         co_return py::cast(result);
    //     };

    //     return std::make_shared<CoroAwaitable>(std::move(convert));
    // });

    // py::class_<LangChainTemplateNodeCpp, LLMNodeBase, std::shared_ptr<LangChainTemplateNodeCpp>>(
    //     _module, "LangChainTemplateNodeCpp")
    //     .def(py::init<>([](std::string template_str) {
    //              return std::make_shared<LangChainTemplateNodeCpp>(std::move(template_str));
    //          }),
    //          py::arg("template"))
    //     .def_property_readonly("template",
    //                            [](LangChainTemplateNodeCpp& self) {
    //                                //
    //                                return self.get_template();
    //                            })
    //     .def("get_input_names", &LangChainTemplateNodeCpp::get_input_names)
    //     .def("execute", [](std::shared_ptr<LangChainTemplateNodeCpp> self, std::shared_ptr<LLMContext> context) {
    //         auto convert = [self](std::shared_ptr<LLMContext> context) -> Task<mrc::pymrc::PyHolder> {
    //             auto result = co_await self->execute(context);

    //             // Convert the output to a python object
    //             co_return py::cast(result);
    //         };

    //         return std::make_shared<CppToPyAwaitable>(convert(context));
    //     });

    _module.attr("__version__") =
        MRC_CONCAT_STR(morpheus_VERSION_MAJOR << "." << morpheus_VERSION_MINOR << "." << morpheus_VERSION_PATCH);
}
}  // namespace morpheus::llm
