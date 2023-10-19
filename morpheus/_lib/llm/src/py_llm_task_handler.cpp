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

#include "py_llm_task_handler.hpp"

#include "pycoro/pycoro.hpp"

#include "morpheus/llm/llm_context.hpp"

#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <coroutine>

namespace morpheus::llm {
namespace py = pybind11;

PyLLMTaskHandler::~PyLLMTaskHandler() = default;

std::vector<std::string> PyLLMTaskHandler::get_input_names() const
{
    PYBIND11_OVERRIDE_PURE(std::vector<std::string>, LLMTaskHandler, get_input_names);
}

Task<LLMTaskHandler::return_t> PyLLMTaskHandler::try_handle(std::shared_ptr<LLMContext> context)
{
    do
    {
        do
        {
            while (google ::_Check_string* _result =
                       google ::Check_EQImpl(google ::GetReferenceableValue(PyGILState_Check()),
                                             google ::GetReferenceableValue(0),
                                             "PyGILState_Check()"
                                             " "
                                             "=="
                                             " "
                                             "0"))
                google ::LogMessageFatal(
                    "/home/mdemoret/Repos/morpheus/morpheus-dev/morpheus/_lib/llm/src/py_llm_task_handler.cpp",
                    42,
                    google ::CheckOpString(_result))
                        .stream()
                    << "Should not have the GIL when resuming a C++ coroutine";
            pybind11 ::gil_scoped_acquire gil;
            pybind11 ::function override =
                pybind11 ::get_override(static_cast<const LLMTaskHandler*>(this), "try_handle");
            if (override)
            {
                auto o_coro         = override(context);
                auto asyncio_module = pybind11 ::module ::import("asyncio");
                if (!asyncio_module.attr("iscoroutine")(o_coro).cast<bool>())
                {
                    pybind11 ::pybind11_fail(
                        ((std ::ostringstream&)(std ::ostringstream() << "Return value from overriden async function "
                                                                      << "LLMTaskHandler"
                                                                      << "::"
                                                                      << "try_handle"
                                                                      << " did not return a coroutine. Returned: "
                                                                      << pybind11 ::str(o_coro).cast<std ::string>()))
                            .str());
                }
                auto o_task = asyncio_module.attr("create_task")(o_coro);
                mrc ::pymrc ::PyHolder o_result;
                {
                    pybind11 ::gil_scoped_release nogil;
                    o_result = co_await mrc ::pycoro ::PyTaskToCppAwaitable(std ::move(o_task));
                    while (google ::_Check_string* _result =
                               google ::Check_EQImpl(google ::GetReferenceableValue(PyGILState_Check()),
                                                     google ::GetReferenceableValue(0),
                                                     "PyGILState_Check()"
                                                     " "
                                                     "=="
                                                     " "
                                                     "0"))
                        google ::LogMessageFatal(
                            "/home/mdemoret/Repos/morpheus/morpheus-dev/morpheus/_lib/llm/src/py_llm_task_handler.cpp",
                            42,
                            google ::CheckOpString(_result))
                                .stream()
                            << "Should not have the GIL after returning from co_await";
                }
                if (pybind11 ::detail ::cast_is_temporary_value_reference<LLMTaskHandler ::return_t>::value)
                {
                    static pybind11 ::detail ::override_caster_t<LLMTaskHandler ::return_t> caster;
                    co_return pybind11 ::detail ::cast_ref<LLMTaskHandler ::return_t>(std ::move(o_result), caster);
                }
                co_return pybind11 ::detail ::cast_safe<LLMTaskHandler ::return_t>(std ::move(o_result));
            }
        } while (false);
        pybind11 ::pybind11_fail(
            ((std ::ostringstream&)(std ::ostringstream() << "Tried to call pure virtual function \""
                                                          << "LLMTaskHandler"
                                                          << "::"
                                                          << "try_handle"
                                                          << "\""))
                .str());
        ;
    } while (false);
}

}  // namespace morpheus::llm
