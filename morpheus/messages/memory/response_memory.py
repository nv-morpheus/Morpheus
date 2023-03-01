# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging

import cupy as cp

import morpheus._lib.messages as _messages
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.utils import logger as morpheus_logger

logger = logging.getLogger(__name__)


@dataclasses.dataclass(init=False)
class ResponseMemory(TensorMemory, cpp_class=_messages.ResponseMemory):
    """Output memory block holding the results of inference."""

    def get_output(self, name: str):
        """
        Get the Tensor stored in the container identified by `name`. Alias for `ResponseMemory.get_tensor`.

        Parameters
        ----------
        name : str
            Key used to do lookup in tensors dict of message container.

        Returns
        -------
        cupy.ndarray
            Tensors corresponding to name.

        Raises
        ------
        AttributeError
            If output name does not exist in message container.

        """
        return self.get_tensor(name)

    def set_output(self, name: str, tensor: cp.ndarray):
        """
        Update the output tensor identified by `name`. Alias for `ResponseMemory.set_tensor`

        Parameters
        ----------
        name : str
            Key used to do lookup in tensors dict of the container.
        tensor : cupy.ndarray
            Tensor as a CuPy array.

        Raises
        ------
        ValueError
            If the number of rows in `tensor` does not match `count`
        """
        self.set_tensor(name, tensor)


@dataclasses.dataclass(init=False)
class ResponseMemoryProbs(ResponseMemory, cpp_class=_messages.ResponseMemoryProbs):
    """
    Subclass of `ResponseMemory` containng an output tensor named 'probs'.

    Parameters
    ----------
    probs : cupy.ndarray
        Probabilities tensor
    """
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(ResponseMemory.get_output, ResponseMemory.set_output)

    def __new__(cls, *args, **kwargs):
        morpheus_logger.deprecated_message_warning(logger, cls, ResponseMemory)
        return super(ResponseMemory, cls).__new__(cls, *args, **kwargs)

    def __init__(self, count: int, probs: cp.ndarray):
        super().__init__(count, tensors={'probs': probs})


@dataclasses.dataclass(init=False)
class ResponseMemoryAE(ResponseMemory, cpp_class=None):
    """
    Subclass of `ResponseMemory` specific to the AutoEncoder pipeline.

    Parameters
    ----------
    probs : cupy.ndarray
        Probabilities tensor

    user_id : str
        User id the inference was performed against.

    explain_df : pd.Dataframe
        Explainability Dataframe, for each feature a column will exist with a name in the form of: `{feature}_z_loss`
        containing the loss z-score along with `max_abs_z` and `mean_abs_z` columns
    """
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(ResponseMemory.get_output, ResponseMemory.set_output)
    user_id = ""
    explain_df = None

    def __init__(self, count: int, probs: cp.ndarray):
        super().__init__(count, tensors={'probs': probs})
