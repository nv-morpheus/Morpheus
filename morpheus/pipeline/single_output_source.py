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

import asyncio
import collections
import inspect
import logging
import os
import signal
import time
import typing
from abc import ABC
from abc import abstractmethod

import neo
import networkx
import typing_utils
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import MultiMessage
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class SingleOutputSource(SourceStage):
    """
    Subclass of SourceStage for building source stages that generate output for single port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(0, 1)

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:
        return out_pair

    @typing.final
    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        ret_val = self._post_build_single(seg, out_ports_pair[0])

        logger.info("Added source: {}\n  └─> {}".format(str(self), pretty_print_type_name(ret_val[1])))

        return [ret_val]
