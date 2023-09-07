# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import datetime
import time
import typing

import cupy as cp
import mrc
import pandas as pd
from mrc.core import operators as ops

from common.data_models import SnapshotData
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("ransomware-preprocess", modes=[PipelineModes.FIL])
class PreprocessingRWStage(SinglePortStage):
    """
    This class extends PreprocessBaseStage and process the features that aree derived from Appshield data.
    It also arranges the snapshots of Appshield data in a sequential order using provided sliding window.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    feature_columns : typing.List[str]
        List of features needed to be extracted.
    sliding_window: int, default = 3
        Window size to arrange the sanpshots in seequential order.
    """

    def __init__(self, c: Config, feature_columns: typing.List[str], sliding_window: int = 3):

        super().__init__(c)

        self._feature_columns = feature_columns
        self._sliding_window = sliding_window
        self._features_len = len(self._feature_columns)

        # Stateful member to hold unprocessed snapshots.
        self._snapshot_dict: typing.Dict[str, typing.List[SnapshotData]] = {}

        # Padding data to map inference response with input messages.
        self._padding_data = [0 for i in range(self._features_len * sliding_window)]

    @property
    def name(self) -> str:
        return "preprocess-rw"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        """
        return (AppShieldMessageMeta, )


    def _sliding_window_offsets(self, ids: typing.List[int], ids_len: int,
                                window: int) -> typing.List[typing.List[int]]:
        """
        Create snapshot_id's sliding sequence for a given window
        """

        sliding_window_offsets = []

        for start in range(ids_len - (window - 1)):
            stop = start + window
            sequence = ids[start:stop]
            consecutive = sequence == list(range(min(sequence), max(sequence) + 1))
            if consecutive:
                sliding_window_offsets.append((start, stop))

        return sliding_window_offsets

    def _rollover_pending_snapshots(self, source_pid_process: str, snapshots_dict):
        """
        Store the unprocessed snapshots from current run to a stateful member to process them in the next run.
        """

        pending_snapshots = []

        keys_to_keep = list(snapshots_dict.keys())

        if len(snapshots_dict) >= self._sliding_window:
            keys_to_keep = keys_to_keep[1:]

        for key in keys_to_keep:
            pending_snapshot = SnapshotData(key, snapshots_dict[key])
            pending_snapshots.append(pending_snapshot)

        self._snapshot_dict[source_pid_process] = pending_snapshots

    def _merge_curr_and_prev_snapshots(self, snapshots_dict: pd.Series, source_pid_process: str) -> pd.DataFrame:
        """
        Merge current run snapshots with previous unprocessed snapshots.
        """

        prev_pending_snapshots = self._snapshot_dict[source_pid_process]

        # If previous pending snapshots that exists in the memory. Just get them to process in this run.
        for prev_pending_snapshot in prev_pending_snapshots:
            snapshots_dict[prev_pending_snapshot.snapshot_id] = prev_pending_snapshot.data

        # Keep snapshot_ids in order to generate sequence.
        snapshots_dict = dict(sorted(snapshots_dict.items()))

        return snapshots_dict

    def _process_batch(self, x: AppShieldMessageMeta) -> MultiInferenceFILMessage:
        """
        This function is invoked for every source_pid_process.
        It looks for any pending snapshots related to the source and pid process in the memory.
        If there are any unprocessed snapshots in the memory, they are merged with existing snapshots,
        and a series of snapshot features generatd based on the specified sliding window,
        followed by the creation of an inference memory message.
        Current run's unprocessed snapshots will be rolled over to the next.
        """

        snapshot_df = x.df

        ldrmodules_df_path = snapshot_df['ldrmodules_df_path']
        pid_process = snapshot_df['pid_process']
        process_name = snapshot_df['process_name']
        snapshot_id = snapshot_df['snapshot_id']
        timestamp = snapshot_df['timestamp']
        source_pid_process = snapshot_df['source_pid_process']

        # Get only feature columns from the dataframe
        snapshot_id = snapshot_df.snapshot_id.iloc[0]

        snapshot_df = snapshot_df[self._feature_columns]
        snapshot_df_size = len(snapshot_df)
        source_pid_processes = snapshot_df.index

        data_l = []
        sequence_l = []

        for source_pid_process in source_pid_processes:
            snapshots_dict = {}
            snapshots_dict[snapshot_id] = snapshot_df.loc[source_pid_process].tolist()
            # Get if there are any previous pending snapshots.
            if source_pid_process in self._snapshot_dict:
                snapshots_dict = self._merge_curr_and_prev_snapshots(snapshots_dict, source_pid_process)

            curr_and_prev_snapshots_size = len(snapshots_dict)

            # Make a dummy set of data and a dummy sequence.
            # When the number of snapshots received for the pid process is less than the sliding window supplied,
            # this is used. For each input message, this is used to construct inference output.
            data = self._padding_data
            sequence = "dummy"

            if curr_and_prev_snapshots_size >= self._sliding_window:
                data = []
                keys = snapshots_dict.keys()

                max_key = max(keys)
                min_key = min(keys)

                for key in keys:
                    data += snapshots_dict[key]

                sequence = f"{min_key}-{max_key}"

            sequence_l.append(sequence)

            data_l.append(data)

            # Rollover pending snapshots
            self._rollover_pending_snapshots(source_pid_process, snapshots_dict)

        # This column is used to identify whether sequence is genuine or dummy
        snapshot_df['sequence'] = sequence_l
        snapshot_df['ldrmodules_df_path'] = ldrmodules_df_path
        snapshot_df['pid_process'] = pid_process
        snapshot_df['process_name'] = process_name
        snapshot_df['snapshot_id'] = snapshot_id
        snapshot_df['timestamp'] = timestamp
        snapshot_df['source_pid_process'] = source_pid_process

        # Convert data to cupy array
        cp_data = cp.asarray(data_l)

        seg_ids = cp.zeros((snapshot_df_size, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, snapshot_df_size, dtype=cp.uint32)
        seg_ids[:, 2] = self._features_len * 3

        memory = InferenceMemoryFIL(count=snapshot_df_size, input__0=cp_data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage(meta=MessageMeta(df=snapshot_df),
                                                 mess_offset=0,
                                                 mess_count=snapshot_df_size,
                                                 memory=memory,
                                                 offset=0,
                                                 count=snapshot_df_size)
        current_time = datetime.datetime.now()
        print(f"Preprocessing snapshot sequence: {sequence} is completed at time: {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        return infer_message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        node = builder.make_node(self.unique_name, ops.map(self._process_batch))
        builder.make_edge(stream, node)
        stream = node

        return stream, MultiInferenceFILMessage
