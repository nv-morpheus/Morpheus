# Copyright (c) 2021, NVIDIA CORPORATION.
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

import re
import typing

from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.pipeline.output.serialize import SerializeStage


class UserProfSerializeStage(SerializeStage):
    @staticmethod
    def add_predictions(x: MultiMessage):
        # Add predictions to final output
        df = x.meta.df
        arr_shape = x.memory.probs.shape
        if len(arr_shape) > 1 and arr_shape[1] > 1:
            df['probs_min'] = x.memory.probs.get()[:, 0]
            df['probs_max'] = x.memory.probs.get()[:, 1]
        else:
            df['probs'] = x.memory.probs.get()[:, 0]

        return x

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        include_columns = None

        if (self._include_columns is not None and len(self._include_columns) > 0):
            include_columns = re.compile("({})".format("|".join(self._include_columns)))

        exclude_columns = [re.compile(x) for x in self._exclude_columns]

        # add predictions
        stream = input_stream[0].async_map(UserProfSerializeStage.add_predictions, executor=self._pipeline.thread_pool)

        # Convert the messages to rows of strings
        stream = stream.async_map(UserProfSerializeStage.convert_to_json,
                                  executor=self._pipeline.thread_pool,
                                  include_columns=include_columns,
                                  exclude_columns=exclude_columns)
        # Return input unchanged
        return stream, typing.List[str]
