# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import typing

import mrc


@typing.overload
def fn_receives_subscriber(fn_or_signature: inspect.Signature) -> bool:
    ...


@typing.overload
def fn_receives_subscriber(fn_or_signature: typing.Callable) -> bool:
    ...


def fn_receives_subscriber(fn_or_signature: inspect.Signature | typing.Callable) -> bool:
    if isinstance(fn_or_signature, inspect.Signature):
        signature = fn_or_signature
    else:
        signature = inspect.signature(fn_or_signature)

    param_iter = iter(signature.parameters.values())

    try:
        first_param = next(param_iter)
        if first_param.annotation is mrc.Subscriber or first_param.annotation == "mrc.Subscriber":
            return True
    except StopIteration:
        pass

    return False
