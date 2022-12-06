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

import typing
from functools import partial

import cupy as cp
import numpy as np
import mrc

import cudf

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage


@register_stage("pcap-preprocess", modes=[PipelineModes.FIL])
class AbpPcapPreprocessingStage(PreprocessBaseStage):

    def __init__(self, c: Config):
        """
        Pre-processing of PCAP data for Anomalous Behavior Profiling Detection Pipeline

        Parameters
        ----------
        c : Config
            The morpheus config
        """

        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = [
            "ack",
            "psh",
            "rst",
            "syn",
            "fin",
            "ppm",
            "data_len",
            "bpp",
            "all",
            "ackpush/all",
            "rst/all",
            "syn/all",
            "fin/all",
        ]
        assert self._fea_length == len(
            self.features
        ), f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-anomaly"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:
        flags_bin_series = cudf.Series(x.get_meta("flags").to_pandas().apply(lambda x: format(int(x), "05b")))

        df = flags_bin_series.str.findall("[0-1]")

        rename_cols_dct = {0: "ack", 1: "psh", 2: "rst", 3: "syn", 4: "fin"}

        # adding [ack, psh, rst, syn, fin] details from the binary flag
        for col in df.columns:
            rename_col = rename_cols_dct[col]
            df[rename_col] = df[col].astype("int8")

        df = df.drop([0, 1, 2, 3, 4], axis=1)

        df["flags_bin"] = flags_bin_series
        df["timestamp"] = x.get_meta("timestamp").astype("int64")

        def round_time_kernel(timestamp, rollup_time, secs):
            for i, ts in enumerate(timestamp):
                x = ts % secs
                y = 1 - (x / secs)
                delta = y * secs
                rollup_time[i] = ts + delta

        df = df.apply_rows(
            round_time_kernel,
            incols=["timestamp"],
            outcols=dict(rollup_time=np.int64),
            kwargs=dict(secs=60000000),
        )

        df["rollup_time"] = cudf.to_datetime(df["rollup_time"], unit="us").dt.strftime("%Y-%m-%d %H:%M")

        # creating flow_id "src_ip:src_port=dst_ip:dst_port"
        df["flow_id"] = (x.get_meta("src_ip") + ":" + x.get_meta("src_port").astype("str") + "=" +
                         x.get_meta("dest_ip") + ":" + x.get_meta("dest_port").astype("str"))
        agg_dict = {
            "ack": "sum",
            "psh": "sum",
            "rst": "sum",
            "syn": "sum",
            "fin": "sum",
            "data_len": "sum",
            "flow_id": "count",
        }

        df["data_len"] = x.get_meta("data_len").astype("int16")

        # group by operation
        grouped_df = df.groupby(["rollup_time", "flow_id"]).agg(agg_dict)

        # Assumption: Each flow corresponds to a single packet flow
        # Given that the roll-up is on 1 minute, packets-per-minute(ppm)=number of flows
        grouped_df.rename(columns={"flow_id": "ppm"}, inplace=True)
        grouped_df.reset_index(inplace=True)

        # bpp - Bytes per packet per flow. In the absence of data on number-of-packets
        # and the assumption that the flow and packet are the same, bpp=bytes/packets
        grouped_df["bpp"] = grouped_df["data_len"] / grouped_df["ppm"]
        grouped_df["all"] = (grouped_df["ack"] + grouped_df["psh"] + grouped_df["rst"] + grouped_df["syn"] +
                             grouped_df["fin"])

        # ackpush/all - Number of flows with ACK+PUSH flags to all flows
        grouped_df["ackpush/all"] = (grouped_df["ack"] + grouped_df["psh"]) / grouped_df["all"]

        # rst/all - Number of flows with RST flag to all flows
        # syn/all - Number of flows with SYN flag to all flows
        # fin/all - Number of flows with FIN flag to all flows
        for col in ["rst", "syn", "fin"]:
            dst_col = "{}/all".format(col)
            grouped_df[dst_col] = grouped_df[col] / grouped_df["all"]

        # Adding index column to retain the order of input messages.
        df["idx"] = df.index

        df = df[["rollup_time", "flow_id", "idx"]]

        # Joining grouped dataframe entries with input dataframe entries to match input messages count.
        merged_df = df.merge(
            grouped_df,
            left_on=["rollup_time", "flow_id"],
            right_on=["rollup_time", "flow_id"],
            how="left",
            suffixes=("_left", ""),
        )

        merged_df = merged_df.sort_values("idx")

        del df, grouped_df

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(merged_df[fea_cols].to_cupy())
        count = data.shape[0]

        # columns required to be added to input message meta
        req_cols = ["flow_id", "rollup_time"]

        for col in req_cols:
            # TODO: temporary work-around for Issue #286
            x.meta.df[col] = merged_df[col].copy(True)

        del merged_df

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryFIL(count=count, input__0=data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage(
            meta=x.meta,
            mess_offset=x.mess_offset,
            mess_count=x.mess_count,
            memory=memory,
            offset=0,
            count=memory.count,
        )

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(
            AbpPcapPreprocessingStage.pre_process_batch,
            fea_len=self._fea_length,
            fea_cols=self.features,
        )

    def _get_preprocess_node(self, builder: mrc.Builder):
        return _stages.AbpPcapPreprocessingStage(builder, self.unique_name)
