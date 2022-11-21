# Copyright (c) 2022, NVIDIA CORPORATION.
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
from abc import ABC
from collections import OrderedDict

import cupy as cp
import numpy as np
import pandas as pd
import srf

import cudf

from morpheus._lib.type_id import tyepid_to_numpy_str
from morpheus.config import CppConfig
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class PreallocatorMixin(ABC):
    """
    Mixin intented to be added to stages, typically source stages,  which are emitting newly constructed DataFrame or
    MessageMeta instances into the segment. During segment build, if the `_needed_columns` addtribut is not empty an
    additional node will be inserted into the graph after the derived class' node which will perform the allocation.

    The exceptions would be non-source stages like DFP's `DFPFileToDataFrameStage` which are not sources but are
    constructing new Dataframe instances, and `LinearBoundaryIngressStage` which is potentially emitting other message
    types such as MultiMessages and it's various derived messages but it would still be the first stage in the given
    segment emitting the message.
    """

    def set_needed_columns(self, needed_columns: OrderedDict):
        """
        Sets the columns needed to perform preallocation. This should only be called by the Pipeline at build time.
        The needed_columns shoudl contain the entire set of columns needed by any other stage in this segment.
        """
        self._needed_columns = needed_columns

    def _preallocate_df(self, df: typing.Union[pd.DataFrame, cudf.DataFrame]):
        # Using a list-comprehension in order to preserve the order
        # Doing `missing_columns = self._needed_columns.keys() - df.columns` loses the order
        missing_columns = [col for col in self._needed_columns.keys() if col not in df.columns]
        if len(missing_columns) > 0:
            if isinstance(df, cudf.DataFrame):
                alloc_func = cp.zeros
            else:
                alloc_func = np.zeros

            num_rows = len(df)
            for column_name in missing_columns:
                column_type = tyepid_to_numpy_str(self._needed_columns[column_name])
                logger.debug("Preallocating column %s[%s]", column_name, column_type)
                df[column_name] = alloc_func(num_rows, column_type)

        return df

    def _preallocate_meta(self, msg: MessageMeta):
        self._preallocate_df(msg.df)
        return msg

    def _preallocate_multi(self, msg: MultiMessage):
        self._preallocate_df(msg.meta.df)
        return msg

    def _post_build_single(self, builder: srf.Builder, out_pair: StreamPair) -> StreamPair:
        (out_stream, out_type) = out_pair
        pretty_type = pretty_print_type_name(out_type)
        logger.info("Added source: {}\n  └─> {}".format(str(self), pretty_type))

        if len(self._needed_columns) > 0:
            node_name = f"{self.unique_name}-preallocate"

            if issubclass(out_type, (MessageMeta, MultiMessage)):
                # Intentionally not using `_build_cpp_node` because `LinearBoundaryIngressStage` lacks a C++ impl
                if CppConfig.get_should_use_cpp():
                    import morpheus._lib.stages as _stages
                    needed_columns = list(self._needed_columns.items())
                    if issubclass(out_type, MessageMeta):
                        stream = _stages.PreallocateMessageMetaStage(builder, node_name, needed_columns)
                    else:
                        stream = _stages.PreallocateMultiMessageStage(builder, node_name, needed_columns)
                else:
                    if issubclass(out_type, MessageMeta):
                        stream = builder.make_node(node_name, self._preallocate_meta)
                    else:
                        stream = builder.make_node(node_name, self._preallocate_multi)
            elif issubclass(out_type, (cudf.DataFrame, pd.DataFrame)):
                stream = builder.make_node(node_name, self._preallocate_df)
            else:
                msg = ("Additional columns were requested to be inserted into the Dataframe, but the output type {}"
                       " isn't a supported type".format(pretty_type))
                raise RuntimeError(msg)

            builder.make_edge(out_stream, stream)
            out_stream = stream

        return (out_stream, out_type)
