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
from functools import partial

import cupy as cp
import srf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryAE
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.stages.inference.auto_encoder_inference_stage import MultiInferenceAEMessage
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
        return (MultiAEMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(x: MultiAEMessage, fea_len: int,
                          feature_columns: typing.List[str]) -> MultiInferenceAEMessage:
        """
        This function performs pre-processing for autoencoder.

        Parameters
        ----------
        x : morpheus.pipeline.preprocess.autoencoder.MultiAEMessage
            Input rows received from Deserialized stage.

        Returns
        -------
        morpheus.pipeline.inference.inference_ae.MultiInferenceAEMessage
            Autoencoder inference message.

        """

        meta_df = x.get_meta(x.meta.df.columns.intersection(feature_columns))
        autoencoder = x.model
        scores_mean = x.train_scores_mean
        scores_std = x.train_scores_std
        count = len(meta_df.index)
        mess_count = count
        input = cp.zeros(meta_df.shape, dtype=cp.float32)

        memory = None

        if autoencoder is not None:
            data = autoencoder.prepare_df(meta_df)
            input = autoencoder.build_input_tensor(data)
            input = cp.asarray(input.detach())
            count = input.shape[0]
            mess_count = x.mess_count

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        memory = InferenceMemoryAE(count=count, input=input, seq_ids=seg_ids)

        infer_message = MultiInferenceAEMessage(meta=x.meta,
                                                mess_offset=x.mess_offset,
                                                mess_count=mess_count,
                                                memory=memory,
                                                offset=0,
                                                count=count,
                                                model=autoencoder,
                                                train_scores_mean=scores_mean,
                                                train_scores_std=scores_std)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessAEStage.pre_process_batch,
                       fea_len=self._fea_length,
                       feature_columns=self._feature_columns)

    def _get_preprocess_node(self, builder: srf.Builder):
        raise NotImplementedError("No C++ node for AE")
