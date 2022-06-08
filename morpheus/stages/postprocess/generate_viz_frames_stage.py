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

import json
import os
import shutil
import typing
import warnings

import srf
import numpy as np
import pandas as pd

from morpheus.config import Config
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class GenerateVizFramesStage(SinglePortStage):
    """
    Class to generate visualization frames.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    out_dir : str
        Output directory to write visualization frames.
    overwrite : bool
        Overwrite file if exists.

    """

    def __init__(self, c: Config, out_dir: str = "./viz_frames", overwrite: bool = False):
        super().__init__(c)

        self._out_dir = out_dir
        self._overwrite = overwrite

        if (os.path.exists(self._out_dir)):
            if (self._overwrite):
                shutil.rmtree(self._out_dir)
            elif (len(list(os.listdir(self._out_dir))) > 0):
                warnings.warn(("Viz output directory '{}' already exists. "
                               "Errors will occur if frames try to be written over existing files. "
                               "Suggest emptying the directory or setting `overwrite=True`").format(self._out_dir))

        os.makedirs(self._out_dir, exist_ok=True)

        self._first_timestamp = -1

    @property
    def name(self) -> str:
        return "gen_viz"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    @staticmethod
    def round_to_sec(x):
        """
        Round to even seconds second.

        Parameters
        ----------
        x : int/float
            Rounding up the value.

        Returns
        -------
        int
            Value rounded up.

        """
        return int(round(x / 1000.0) * 1000)

    def _to_vis_df(self, x: MultiResponseProbsMessage):

        idx2label = {
            0: 'address',
            1: 'bank_acct',
            2: 'credit_card',
            3: 'email',
            4: 'govt_id',
            5: 'name',
            6: 'password',
            7: 'phone_num',
            8: 'secret_keys',
            9: 'user'
        }

        df = x.get_meta(["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "data"])

        def indent_data(y: str):
            try:
                return json.dumps(json.loads(y), indent=3)
            except:  # noqa: E722
                return y

        df["data"] = df["data"].apply(indent_data)

        pass_thresh = (x.probs >= 0.5).any(axis=1)
        max_arg = x.probs.argmax(axis=1)

        condlist = [pass_thresh]

        choicelist = [max_arg]

        index_sens_info = np.select(condlist, choicelist, default=len(idx2label))

        df["si"] = pd.Series(np.choose(index_sens_info.get(), list(idx2label.values()) + ["none"]).tolist())

        df["ts_round_sec"] = (df["timestamp"] / 1000.0).astype(int) * 1000

        # Return a list of tuples of (ts_round_sec, dataframe)
        return [(key, group) for key, group in df.groupby(df.ts_round_sec)]

    def _write_viz_file(self, x: typing.List[typing.Tuple[int, pd.DataFrame]]):

        curr_timestamp = x[0][0]

        in_df = pd.concat([df for _, df in x], ignore_index=True).sort_values(by=["timestamp"])

        # curr_timestamp = GenerateVizFramesStage.round_to_sec(in_df["timestamp"].iloc[0])

        if (self._first_timestamp == -1):
            self._first_timestamp = curr_timestamp

        offset = (curr_timestamp - self._first_timestamp) / 1000

        fn = os.path.join(self._out_dir, "{}.csv".format(offset))

        assert not os.path.exists(fn)

        in_df.to_csv(fn, columns=["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "si", "data"])

    def _build_single(self, seg: srf.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Convert stream to dataframes
        stream = stream.map(self._to_vis_df)  # Convert group to dataframe

        # Flatten the list of tuples
        stream = stream.flatten()

        # Partition by group times
        stream = stream.partition(10000, timeout=10, key=lambda x: x[0])  # Group
        # stream = stream.filter(lambda x: len(x) > 0)

        stream.sink(self._write_viz_file)

        # Return input unchanged
        return input_stream
