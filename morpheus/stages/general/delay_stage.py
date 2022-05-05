# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import logging
import typing
from functools import reduce

import cupy as cp
import neo
from neo.core import operators as ops
from tqdm import TMonitor
from tqdm import TqdmSynchronisationWarning
from tqdm import tqdm

import cudf

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline import Stage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.logging import deprecated_stage_warning

logger = logging.getLogger(__name__)

class DelayStage(SinglePortStage):
    """
    Delay stage class. Used to buffer all inputs until the timeout duration is hit. At that point all messages
    will be dumped into downstream stages. Useful for testing performance of one stage at a time.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config, duration: str):
        super().__init__(c)

        self._duration = duration

    @property
    def name(self) -> str:
        return "delay"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        # This stage is no longer needed and is just a pass thru stage
        deprecated_stage_warning(logger, type(self), self.unique_name)

        return input_stream
