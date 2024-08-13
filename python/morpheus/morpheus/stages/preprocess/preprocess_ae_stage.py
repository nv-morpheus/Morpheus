# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
from functools import partial

import cupy as cp
import mrc

import morpheus._lib.messages as _messages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage

logger = logging.getLogger(__name__)


@register_stage("preprocess", modes=[PipelineModes.AE])
class PreprocessAEStage(PreprocessBaseStage):
    """
    Prepare Autoencoder input DataFrames for inference.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length
        self._feature_columns = c.ae.feature_columns

    @property
    def name(self) -> str:
        return "preprocess-ae"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        """
        return (ControlMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(msg: ControlMessage, fea_len: int, feature_columns: typing.List[str]) -> ControlMessage:
        """
        This function performs pre-processing for autoencoder.

        Parameters
        ----------
        msg : morpheus.messages.ControlMessage
            Input rows received from Deserialized stage.
        fea_len : int
            Number of input features.
        feature_columns : typing.List[str]
            List of feature columns.

        Returns
        -------
        morpheus.messages.ControlMessage

        """
        meta_df = msg.payload().get_data(msg.payload().df.columns.intersection(feature_columns))

        autoencoder = msg.get_metadata("model")
        scores_mean = msg.get_metadata("train_scores_mean")
        scores_std = msg.get_metadata("train_scores_std")
        count = len(meta_df.index)

        inputs = cp.zeros(meta_df.shape, dtype=cp.float32)

        if autoencoder is not None:
            data = autoencoder.prepare_df(meta_df)
            inputs = autoencoder.build_input_tensor(data)
            inputs = cp.asarray(inputs.detach())
            count = inputs.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        msg.set_metadata("model", autoencoder)
        msg.set_metadata("train_scores_mean", scores_mean)
        msg.set_metadata("train_scores_std", scores_std)
        msg.tensors(_messages.TensorMemory(count=count, tensors={"input": inputs, "seq_ids": seg_ids}))
        return msg

    def _get_preprocess_fn(self) -> typing.Callable[[ControlMessage], ControlMessage]:
        return partial(PreprocessAEStage.pre_process_batch,
                       fea_len=self._fea_length,
                       feature_columns=self._feature_columns)

    def _get_preprocess_node(self, builder: mrc.Builder):
        raise NotImplementedError("No C++ node for AE")
