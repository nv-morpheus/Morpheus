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

import cupy as cp
import numpy as np
import pandas as pd
import srf

import cudf

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class SingleOutputSource(_pipeline.SourceStage):
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

    def _post_build_single(self, builder: srf.Builder, out_pair: StreamPair) -> StreamPair:
        return out_pair

    @typing.final
    def _post_build(self, builder: srf.Builder, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        (out_stream, out_type) = self._post_build_single(builder, out_ports_pair[0])
        pretty_type = pretty_print_type_name(out_type)
        logger.info("Added source: {}\n  └─> {}".format(str(self), pretty_type))

        if len(self._needed_columns) > 0:
            node_name = f"{self.unique_name}-preallocate"

            if issubclass(out_type, (MessageMeta, MultiMessage)):
                if self._build_cpp_node():
                    import morpheus._lib.stages as _stages
                    if issubclass(out_type, MessageMeta):
                        stream = _stages.PreallocateMessageMetaStage(builder, node_name, self._needed_columns)
                    else:
                        stream = _stages.PreallocateMultiMessageStage(builder, node_name, self._needed_columns)
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

            logger.info("Added stage: <{}>\n  └─> {} -> {}".format(node_name, pretty_type, pretty_type))

        return [(out_stream, out_type)]

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
                column_type = self._needed_columns[column_name]
                logger.debug("Preallocating column %s[%s]", column_name, column_type)
                df[column_name] = alloc_func(num_rows, column_type)

        return df

    def _preallocate_meta(self, msg: MessageMeta):
        self._preallocate_df(msg.df)
        return msg

    def _preallocate_multi(self, msg: MultiMessage):
        self._preallocate_df(msg.meta.df)
        return msg
