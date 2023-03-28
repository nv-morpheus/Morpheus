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

import typing

import cupy as cp
import mrc
import pandas as pd

from common.data_models import SnapshotData
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage


@register_stage("ransomware-preprocess", modes=[PipelineModes.FIL])
class PreprocessingRWStage(PreprocessBaseStage):
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

    def _sliding_window_offsets(self, ids: typing.List[int], ids_len: int,
                                window: int) -> typing.List[typing.List[int]]:
        """
        Create snapshot_id's sliding sequence for a given window
        """

        sliding_window_offsets = []

        for start in range(ids_len - (window - 1)):
            stop = start + window
            sequence = ids[start:stop]
            consecutive = sorted(sequence) == list(range(min(sequence), max(sequence) + 1))
            if consecutive:
                sliding_window_offsets.append((start, stop))

        return sliding_window_offsets

    def _rollover_pending_snapshots(self,
                                    snapshot_ids: typing.List[int],
                                    source_pid_process: str,
                                    snapshot_df: pd.DataFrame):
        """
        Store the unprocessed snapshots from current run to a stateful member to process them in the next run.
        """

        pending_snapshots = []

        for snapshot_id in snapshot_ids[1 - self._sliding_window:]:
            pending_snapshot_data = snapshot_df[snapshot_df.index == snapshot_id].values[0]
            pending_snapshot = SnapshotData(snapshot_id, pending_snapshot_data)
            pending_snapshots.append(pending_snapshot)

        if pending_snapshots:
            self._snapshot_dict[source_pid_process] = pending_snapshots

    def _merge_curr_and_prev_snapshots(self, snapshot_df: pd.DataFrame, source_pid_process: str) -> pd.DataFrame:
        """
        Merge current run snapshots with previous unprocessed snapshots.
        """

        prev_pending_snapshots = self._snapshot_dict[source_pid_process]

        # If previous pending snapshots that exists in the memory. Just get them to process in this run.
        for prev_pending_snapshot in prev_pending_snapshots:
            snapshot_df.loc[prev_pending_snapshot.snapshot_id] = prev_pending_snapshot.data

        # Keep snapshot_ids in order to generate sequence.
        snapshot_df = snapshot_df.sort_index()

        return snapshot_df

    def _pre_process_batch(self, x: MultiMessage) -> MultiInferenceFILMessage:
        """
        This function is invoked for every source_pid_process.
        It looks for any pending snapshots related to the source and pid process in the memory.
        If there are any unprocessed snapshots in the memory, they are merged with existing snapshots,
        and a series of snapshot features generatd based on the specified sliding window,
        followed by the creation of an inference memory message.
        Current run's unprocessed snapshots will be rolled over to the next.
        """

        snapshot_df = x.get_meta()
        curr_snapshots_size = len(snapshot_df)

        # Set snapshot_id as index this is used to get ordered snapshots based on sliding window.
        snapshot_df.index = snapshot_df.snapshot_id

        # Get source_pid_process.
        source_pid_process = snapshot_df.source_pid_process.iloc[0]

        # Get only feature columns from the dataframe
        snapshot_df = snapshot_df[self._feature_columns]

        # Get if there are any previous pending snapshots.
        if source_pid_process in self._snapshot_dict:
            snapshot_df = self._merge_curr_and_prev_snapshots(snapshot_df, source_pid_process)

        snapshot_ids = snapshot_df.index.values

        curr_and_prev_snapshots_size = len(snapshot_df)

        # Make a dummy set of data and a dummy sequence.
        # When the number of snapshots received for the pid process is less than the sliding window supplied,
        # this is used. For each input message, this is used to construct inference output.
        data = [self._padding_data] * curr_snapshots_size
        sequence = ["dummy"] * curr_snapshots_size

        if curr_and_prev_snapshots_size >= self._sliding_window:
            # Rollover and current snapshots are used to generate sliding window offsets
            offsets = self._sliding_window_offsets(snapshot_ids,
                                                   curr_and_prev_snapshots_size,
                                                   window=self._sliding_window)

            # Generate data from the sliding window offsets.
            for start, stop in offsets:
                data[start] = list(snapshot_df[start:stop].values.ravel())
                sequence[start] = str(snapshot_df.index[start]) + "-" + str(snapshot_df.index[stop - 1])

        # Rollover pending snapshots
        self._rollover_pending_snapshots(snapshot_ids, source_pid_process, snapshot_df)

        # This column is used to identify whether sequence is genuine or dummy
        x.set_meta('sequence', sequence)

        # Convert data to cupy array
        data = cp.asarray(data)

        seg_ids = cp.zeros((curr_snapshots_size, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(x.mess_offset, x.mess_offset + curr_snapshots_size, dtype=cp.uint32)
        seg_ids[:, 2] = self._features_len * 3

        memory = InferenceMemoryFIL(count=curr_snapshots_size, input__0=data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage.from_message(x, memory=memory)
        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pre_process_batch_fn = self._pre_process_batch
        return pre_process_batch_fn

    def _get_preprocess_node(self, builder: mrc.Builder):
        raise NotImplementedError("No C++ node supported")
