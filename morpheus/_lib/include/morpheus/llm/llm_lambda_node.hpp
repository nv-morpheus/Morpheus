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

#include "morpheus/export.h"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node_base.hpp"
#include "morpheus/types.hpp"

#include <boost/type_traits/function_traits.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/type_traits.hpp>
#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace morpheus::llm {

template <typename T>
struct closure_traits : closure_traits<decltype(&T::operator())>
{};

#define REM_CTOR(...) __VA_ARGS__
#define SPEC(cv, var, is_var)                                                        \
    template <typename C, typename R, typename... Args>                              \
    struct closure_traits<R (C::*)(Args... REM_CTOR var) cv>                         \
    {                                                                                \
        using arity       = std::integral_constant<std::size_t, sizeof...(Args)>;    \
        using is_variadic = std::integral_constant<bool, is_var>;                    \
        using is_const    = std::is_const<int cv>;                                   \
                                                                                     \
        using result_type = R;                                                       \
                                                                                     \
        template <std::size_t i>                                                     \
        using arg = typename std::tuple_element<i, std::tuple<Args..., void>>::type; \
    };

SPEC(const, (, ...), 1)
SPEC(const, (), 0)
SPEC(, (, ...), 1)
SPEC(, (), 0)

template <typename Callable>
using return_type_of_t = typename decltype(std::function{std::declval<Callable>()})::result_type;

template <typename T>
struct ExtractValueType
{
    using value_t = T;
};

template <template <typename, typename...> class ClassT, typename T, typename... ArgsT>
struct ExtractValueType<ClassT<T, ArgsT...>>
{
    using value_t = T;
};

template <typename ReturnT, typename... ArgsT>
class LLMLambdaNodeBase : public LLMNodeBase
{
  public:
    using function_t = std::function<Task<ReturnT>(ArgsT...)>;

    LLMLambdaNodeBase(std::vector<std::string> input_names, function_t function) :
      m_input_names(std::move(input_names)),
      m_function(std::move(function))
    {}

    std::vector<std::string> get_input_names() const override
    {
        return m_input_names;
    }

  protected:
    std::vector<std::string> m_input_names;
    function_t m_function;
};

template <typename ReturnT, typename... ArgsT>
class LLMLambdaNode : public LLMLambdaNodeBase<ReturnT, ArgsT...>
{
  public:
    using base_t = LLMLambdaNodeBase<ReturnT, ArgsT...>;
    using typename base_t::function_t;

    // Copy the constructor
    LLMLambdaNode(std::vector<std::string> input_names, std::function<Task<ReturnT>(ArgsT...)> function) :
      base_t(std::move(input_names), std::move(function))
    {}

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
    {
        using args_tuple_t = std::tuple<ArgsT...>;

        if constexpr (std::tuple_size<args_tuple_t>::value == 0)
        {
            auto outputs = co_await this->m_function();

            nlohmann::json outputs_json = std::move(outputs);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
        else if constexpr (std::tuple_size<args_tuple_t>::value == 1)
        {
            const auto& arg = context->get_input();

            auto output = co_await this->m_function(arg.get<std::tuple_element_t<0, args_tuple_t>>());

            nlohmann::json outputs_json = std::move(output);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
        else
        {
            auto args = context->get_inputs();

            auto outputs = co_await this->m_function(args);

            nlohmann::json outputs_json = std::move(outputs);

            // Set the outputs
            context->set_output(std::move(outputs_json));

            co_return context;
        }
    }
};

// template <typename ReturnT, typename SingleArgT>
// class LLMLambdaNode<ReturnT, SingleArgT> : public LLMLambdaNodeBase<ReturnT, SingleArgT>
// {
//   public:
//     using base_t = LLMLambdaNodeBase<ReturnT, SingleArgT>;
//     using typename base_t::function_t;

//     // Copy the constructor
//     LLMLambdaNode(std::vector<std::string> input_names, std::function<Task<ReturnT>(SingleArgT)> function) :
//       base_t(std::move(input_names), std::move(function))
//     {}

//     Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override
//     {
//         const auto& arg = context->get_input();

//         auto output = co_await this->m_function(arg.get<SingleArgT>());

//         nlohmann::json outputs_json = std::move(output);

//         // Set the outputs
//         context->set_output(std::move(outputs_json));

//         co_return context;
//     }
// };

// template <typename ReturnT, typename... ArgsT>
// LLMLambdaNode(std::vector<std::string>, std::function<ReturnT(ArgsT...)>&&)
//     -> LLMLambdaNode<typename ExtractValueType<ReturnT>::value_t, ArgsT...>;
// template <std::size_t... Is>
// auto make_argument_names(std::index_sequence<Is...>)
// {
//     return std::vector<std::string>{std::string{"arg"} + std::to_string(Is)...};
// }

template <typename ReturnT, typename... ArgsT>
auto make_lambda_node(std::function<ReturnT(ArgsT...)>&& fn)
{
    using function_t = std::function<ReturnT(ArgsT...)>;

    using return_t = typename ExtractValueType<typename function_t::result_type>::value_t;

    auto make_args = []<std::size_t... Is>(std::index_sequence<Is...>)
    {
        return std::vector<std::string>{std::string{"arg"} + std::to_string(Is)...};
    };

    return std::make_shared<LLMLambdaNode<return_t, ArgsT...>>(make_args(std::index_sequence_for<ArgsT...>{}),
                                                               std::move(fn));
}

template <typename FuncT>
auto make_lambda_node(FuncT&& fn)
{
    return make_lambda_node(std::function{std::forward<FuncT>(fn)});
    // using fn_traits_t = closure_traits<FuncT>;

    // // static_assert(mrc::is_base_of_template<Task, typename fn_traits_t::result_type>::value,
    // //               "Return type must be a Task");

    // // Get the value of the task return type
    // using return_t = typename ExtractValueType<typename fn_traits_t::result_type>::value_t;

    // // using node_t = std::remove_pointer_t<decltype(new LLMLambdaNode({}, std::function{std::forward<FuncT>(fn)}))>;

    // auto* new_ptr = new LLMLambdaNode({}, std::function{std::forward<FuncT>(fn)});

    // return new_ptr;

    // return std::shared_ptr(new_ptr);
    // return new LLMLambdaNode(std::vector<std::string>{}, std::function{std::forward<FuncT>(fn)});
}

}  // namespace morpheus::llm
