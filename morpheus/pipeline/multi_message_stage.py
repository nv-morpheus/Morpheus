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
import time

import srf

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


def _get_time_ms():
    return round(time.time() * 1000)


class MultiMessageStage(_pipeline.SinglePortStage):
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

    def _post_build_single(self, seg: srf.Builder, out_pair: StreamPair) -> StreamPair:

        # Check if we are debug and should log timestamps. Disable for C++ nodes
        if (self._config.debug and self._should_log_timestamps and not self._build_cpp_node()):
            # Cache the name property. Removes dependency on self in callback
            cached_name = self.name

            logger.info("Adding timestamp info for stage: '%s'", cached_name)

            def post_timestamps(x: MultiMessage):

                curr_time = _get_time_ms()

                x.set_meta("_ts_" + cached_name, curr_time)

                # Must return the original object
                return x

            # Only have one port
            post_ts = seg.make_node(self.unique_name + "-ts", post_timestamps)
            seg.make_edge(out_pair[0], post_ts)

            # Keep the type unchanged
            out_pair = (post_ts, out_pair[1])

        return super()._post_build_single(seg, out_pair)
