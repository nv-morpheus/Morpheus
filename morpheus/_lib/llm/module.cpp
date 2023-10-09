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

// namespace pybind11::detail {
// template <typename Iterator, typename SFINAE = decltype((*std::declval<Iterator&>()).key())>
// class iterator_key_access
// {
//   private:
//     using pair_type = decltype(*std::declval<Iterator&>());

//   public:
//     /* If either the pair itself or the element of the pair is a reference, we
//      * want to return a reference, otherwise a value. When the decltype
//      * expression is parenthesized it is based on the value category of the
//      * expression; otherwise it is the declared type of the pair member.
//      * The use of declval<pair_type> in the second branch rather than directly
//      * using *std::declval<Iterator &>() is a workaround for nvcc
//      * (it's not used in the first branch because going via decltype and back
//      * through declval does not perfectly preserve references).
//      */
//     using result_type = std::conditional_t<std::is_reference<decltype(*std::declval<Iterator&>())>::value,
//                                            decltype(((*std::declval<Iterator&>()).first)),
//                                            decltype(std::declval<pair_type>().first)>;
//     result_type operator()(Iterator& it) const
//     {
//         return (*it).key();
//     }
// };
// }  // namespace pybind11::detail

namespace morpheus::llm {
namespace py = pybind11;

PYBIND11_MODULE(llm, _module)
{
    py::print("Loading");

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
    //     [](const LLMTask& self) {
    //         return py::detail::make_iterator_impl<iterator_key_access<typename Iterator>,
    //                                               return_value_policy Policy,
    //                                               typename Iterator,
    //                                               typename Sentinel,
    //                                               typename ValueType>(self.task_dict.items().begin(),
    //                                                                   self.task_dict.items().end());
    //         self.task_dict.items().begin().begin() return py::make_key_iterator(map.begin(), map.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "items",
    //     [](const LLMTask& self) {
    //         return py::make_iterator(self.begin(), self.end());
    //     },
    //     py::keep_alive<0, 1>())
    // .def(
    //     "values",
    //     [](const LLMTask& map) {
    //         return py::make_value_iterator(self., self.end());
    //     },
    //     py::keep_alive<0, 1>());

    py::class_<LLMContext, std::shared_ptr<LLMContext>>(_module, "LLMContext")
        .def_property_readonly("name",
                               [](LLMContext& self) {
                                   return self.name();
                               })
        .def_property_readonly("full_name",
                               [](LLMContext& self) {
                                   return self.full_name();
                               })
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
        .def("task",
             [](LLMContext& self) {
                 return self.task();
             })
        .def("message",
             [](LLMContext& self) {
                 return self.message();
             })
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
