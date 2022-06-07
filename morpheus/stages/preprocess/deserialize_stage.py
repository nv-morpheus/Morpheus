# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import typing
from functools import partial

import neo
from neo.core import operators as ops

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class DeserializeStage(MultiMessageStage):
    """
    This stage deserialize the output of `FileSourceStage`/`KafkaSourceStage` into a `MultiMessage`. This
    should be one of the first stages after the `Source` object.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._max_concurrent = c.num_threads

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MessageMeta)

    def supports_cpp_node(self):
        # Enable support by default
        return True

    @staticmethod
    def process_dataframe(x: MessageMeta, batch_size: int) -> typing.List[MultiMessage]:
        """
        The deserialization of the cudf is implemented in this function.

        Parameters
        ----------
        x : cudf.DataFrame
            Input rows that needs to be deserilaized.
        batch_size : int
            Batch size.

        """

        full_message = MultiMessage(meta=x, mess_offset=0, mess_count=x.count)

        # Now break it up by batches
        output = []

        for i in range(0, full_message.mess_count, batch_size):
            output.append(full_message.get_slice(i, min(i + batch_size, full_message.mess_count)))

        return output

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        def node_fn(input: neo.Observable, output: neo.Subscriber):

            input.pipe(ops.map(partial(DeserializeStage.process_dataframe, batch_size=self._batch_size)),
                       ops.flatten()).subscribe(output)

        if self._build_cpp_node():
            stream = neos.DeserializeStage(seg, self.unique_name, self._batch_size)
        else:
            stream = seg.make_node_full(self.unique_name, node_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type
