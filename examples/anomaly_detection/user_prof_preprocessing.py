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

import typing
from functools import partial

import cudf
import numpy as np
import cupy as cp
from morpheus.config import Config
from morpheus.pipeline.preprocessing import PreprocessBaseStage
from morpheus.pipeline.messages import (
    InferenceMemoryFIL,
    MultiInferenceFILMessage,
    MultiInferenceMessage,
    MultiMessage,
)


class UserProfPreprocessingStage(PreprocessBaseStage):
    def __init__(self, c: Config):
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

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:
        x.meta.df["flags_bin"] = x.meta.df["flags"].apply(lambda x: format(int(x), "05b"))
        x.meta.df = cudf.from_pandas(x.meta.df)
        flag_split_df = x.meta.df["flags_bin"].str.findall("[0-1]")
        rename_cols_dct = {0: "ack", 1: "psh", 2: "rst", 3: "syn", 4: "fin"}

        # adding [ack, psh, rst, syn, fin] details from the binary flag
        for col in flag_split_df.columns:
            rename_col = rename_cols_dct[col]
            x.meta.df[rename_col] = flag_split_df[col].astype("int8")

        x.meta.df["timestamp"] = x.meta.df["timestamp"].astype("int64")

        def round_to_minute_kernel(timestamp, rollup, kwarg1):
            for i, ts in enumerate(timestamp):
                x = ts % 60000000
                y = 1 - (x / 60000000)
                delta = y * 60000000
                rollup[i] = ts + delta

        x.meta.df = x.meta.df.apply_rows(
            round_to_minute_kernel,
            incols=["timestamp"],
            outcols=dict(rollup=np.int64),
            kwargs=dict(kwarg1=0),
        )
        x.meta.df["rollup"] = cudf.to_datetime(x.meta.df["rollup"], unit="us")
        x.meta.df["rollup"] = x.meta.df["rollup"].dt.strftime("%Y-%m-%d %H:%M")

        # creating flow_id "src_ip:src_port=dst_ip:dst_port"
        x.meta.df["flow_id"] = (x.meta.df["src_ip"] + ":" + x.meta.df["src_port"].astype("str") + "="
                                + x.meta.df["dest_ip"] + ":" + x.meta.df["dest_port"].astype("str"))

        agg_dict = {
            "ack": "sum",
            "psh": "sum",
            "rst": "sum",
            "syn": "sum",
            "fin": "sum",
            "data_len": "sum",
            "flow_id": "count",
        }

        x.meta.df["data_len"] = x.meta.df["data_len"].astype("int16")

        # group by operation
        x.meta.df = x.meta.df.groupby(["rollup", "flow_id"]).agg(agg_dict)

        # Assumption: Each flow corresponds to a single packet flow
        # Given that the roll-up is on 1 minute, packets-per-minute(ppm)=number of flows
        x.meta.df.rename(columns={"flow_id": "ppm"}, inplace=True)
        x.meta.df.reset_index(inplace=True)

        # bpp - Bytes per packet per flow. In the absence of data on number-of-packets
        # and the assumption that the flow and packet are the same, bpp=bytes/packets
        x.meta.df["bpp"] = x.meta.df["data_len"] / x.meta.df["ppm"]
        x.meta.df["all"] = (x.meta.df["ack"] + x.meta.df["psh"] + x.meta.df["rst"] + x.meta.df["syn"]
                            + x.meta.df["fin"])
        # ackpush/all - Number of flows with ACK+PUSH flags to all flows
        x.meta.df["ackpush/all"] = (x.meta.df["ack"] + x.meta.df["psh"]) / x.meta.df["all"]

        # rst/all - Number of flows with RST flag to all flows
        # syn/all - Number of flows with SYN flag to all flows
        # fin/all - Number of flows with FIN flag to all flows
        for col in ["rst", "syn", "fin"]:
            dst_col = "{}/all".format(col)
            x.meta.df[dst_col] = x.meta.df[col] / x.meta.df["all"]

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(x.meta.df[fea_cols].as_gpu_matrix(order="C"))

        x.meta.df = x.meta.df[["rollup", "flow_id", "data_len"]].to_pandas()
        count = data.shape[0]

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
            UserProfPreprocessingStage.pre_process_batch,
            fea_len=self._fea_length,
            fea_cols=self.features,
        )
