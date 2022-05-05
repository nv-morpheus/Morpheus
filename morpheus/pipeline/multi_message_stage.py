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
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)

def _get_time_ms():
    return round(time.time() * 1000)

class MultiMessageStage(SinglePortStage):
    """
    Subclass of SinglePortStage with option to log timestamps in MessageMeta dataframe.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):

        # Derived classes should set this to true to log timestamps in debug builds
        self._should_log_timestamps = False

        super().__init__(c)

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        # Check if we are debug and should log timestamps
        if (self._config.debug and self._should_log_timestamps):
            # Cache the name property. Removes dependency on self in callback
            cached_name = self.name

            logger.info("Adding timestamp info for stage: '%s'", cached_name)

            def post_timestamps(x: MultiMessage):

                curr_time = _get_time_ms()

                x.set_meta("_ts_" + cached_name, curr_time)

            # Only have one port
            post_ts = seg.make_node(self.unique_name + "-ts", post_timestamps)
            seg.make_edge(out_pair[0], post_ts)

            # Keep the type unchanged
            out_pair = (post_ts, out_pair[1])

        return super()._post_build_single(seg, out_pair)
