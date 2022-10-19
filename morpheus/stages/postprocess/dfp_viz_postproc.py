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
import os
import glob

import numpy as np
import pandas as pd
import srf
import srf.core.operators as ops

import morpheus._lib.stages as _stages
from morpheus._lib.file_types import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta

from morpheus.io import serializers

logger = logging.getLogger(__name__)


@register_stage("dfp-viz-postproc", modes=[PipelineModes.AE])
class DFPVizPostprocStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 period: str = "D",
                 overwrite: bool = True,
                 output_dir: str = ".",
                 prefix: str = "dfp-viz-"):
        super().__init__(c)

        self._user_column_name = c.ae.userid_column_name
        self._timestamp_column = c.ae.timestamp_column_name
        self._feature_columns = c.ae.feature_columns
        self._period = period
        self._overwrite = overwrite
        self._file_type = FileTypes.CSV
        self._output_dir = output_dir
        self._prefix = prefix
        self._output_filenames = []

        if (self._overwrite):
            output_glob = os.path.join(self._output_dir, self._prefix + "*")
            fileList = glob.glob(output_glob)
            for filePath in fileList:
                try:
                    os.remove(filePath)
                except:
                    print("Error while deleting file : ", filePath)

    @property
    def name(self) -> str:
        return "dfp-viz-postproc"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiAEMessage`, ]
            Accepted input types.

        """
        return (MultiAEMessage, )

    def supports_cpp_node(self):
        return False

    def _convert_to_strings(self, df: pd.DataFrame, include_header=False):
        if (self._file_type == FileTypes.JSON):
            output_strs = serializers.df_to_json(df)
        elif (self._file_type == FileTypes.CSV):
            output_strs = serializers.df_to_csv(df, include_header=include_header, include_index_col=False)
        else:
            raise NotImplementedError("Unknown file type: {}".format(self._file_type))

        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]

        return output_strs

    def _postprocess(self, x: MultiAEMessage):

        viz_pdf = pd.DataFrame()
        viz_pdf["user"] = x.get_meta(self._user_column_name)
        viz_pdf["time"] = x.get_meta(self._timestamp_column)
        datetimes = pd.to_datetime(viz_pdf["time"], errors='coerce')
        viz_pdf["period"] = datetimes.dt.to_period(self._period)

        for f in self._feature_columns:
            viz_pdf[f + "_score"] =  x.get_meta(f + "_z_loss")

        viz_pdf["anomalyScore"] = x.get_meta("mean_abs_z")

        return MessageMeta(df=viz_pdf)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):

            def write_to_files(x: MultiAEMessage):

                message_meta = self._postprocess(x)

                unique_periods = message_meta.df["period"].unique()

                for period in unique_periods:
                    period_df = message_meta.df[message_meta.df["period"] == period]
                    period_df = period_df.drop(["period"], axis=1)
                    output_file = os.path.join(self._output_dir, self._prefix + str(period) + ".csv")
                    
                    is_first = False
                    if output_file not in self._output_filenames:
                        self._output_filenames.append(output_file)
                        is_first = True

                    lines = self._convert_to_strings(period_df, include_header=is_first)
                    os.makedirs(os.path.realpath(os.path.dirname(output_file)), exist_ok=True)
                    with open(output_file, "a") as out_file:
                        out_file.writelines(lines)

                return x

            obs.pipe(ops.map(write_to_files)).subscribe(sub)

        dfp_viz_postproc = builder.make_node_full(self.unique_name, node_fn)

        builder.make_edge(stream, dfp_viz_postproc)
        stream = dfp_viz_postproc

        return stream, input_stream[1]