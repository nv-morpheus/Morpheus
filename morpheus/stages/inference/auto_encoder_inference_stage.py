# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
import numpy as np
import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiResponseAEMessage
from morpheus.messages import ResponseMemory
from morpheus.messages import ResponseMemoryAE
from morpheus.messages.multi_inference_ae_message import MultiInferenceAEMessage
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_response_message import MultiResponseProbsMessage
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


class _AutoEncoderInferenceWorker(InferenceWorker):

    def __init__(self, inf_queue: ProducerConsumerQueue, c: Config):
        super().__init__(inf_queue)

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length

        self._feature_columns = c.ae.feature_columns

    def init(self):

        pass

    def build_output_message(self, x: MultiInferenceAEMessage) -> MultiResponseAEMessage:
        """
        Create initial inference response message with result values initialized to zero. Results will be
        set in message as each inference batch is processed.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiInferenceAEMessage`
            Batch of autoencoder inference messages.

        Returns
        -------
        `morpheus.pipeline.messagesMultiResponseAEMessage`
            Response message with autoencoder results calculated from inference results.
        """

        dims = self.calc_output_dims(x)
        output_dims = (x.mess_count, *dims[1:])

        memory = ResponseMemoryAE(count=output_dims[0], probs=cp.zeros(output_dims))

        # Override the default to return the response AE
        output_message = MultiResponseAEMessage(meta=x.meta,
                                                mess_offset=x.mess_offset,
                                                mess_count=x.mess_count,
                                                memory=memory,
                                                offset=x.offset,
                                                count=x.count,
                                                user_id=x.user_id)

        return output_message

    def calc_output_dims(self, x: MultiInferenceAEMessage) -> typing.Tuple:

        # reconstruction loss and zscore
        return (x.count, 2)

    def process(self, batch: MultiInferenceAEMessage, cb: typing.Callable[[ResponseMemory], None]):
        """
        This function processes inference batch by using batch's model to calculate anomaly scores
        and adding results to response.

        Parameters
        ----------
        batch : `morpheus.pipeline.messages.MultiInferenceMessage`
            Batch of inference messages.
        cb : typing.Callable[[`morpheus.pipeline.messages.ResponseMemory`], None]
            Inference callback.

        """
        data = batch.get_meta(batch.meta.df.columns.intersection(self._feature_columns))

        explain_cols = [x + "_z_loss" for x in self._feature_columns] + ["max_abs_z", "mean_abs_z"]
        explain_df = pd.DataFrame(np.empty((batch.count, (len(self._feature_columns) + 2)), dtype=object),
                                  columns=explain_cols)

        if batch.model is not None:
            rloss_scores = batch.model.get_anomaly_score(data)

            results = batch.model.get_results(data, return_abs=True)
            scaled_z_scores = [col for col in results.columns if col.endswith('_z_loss')]
            scaled_z_scores.extend(['max_abs_z', 'mean_abs_z'])
            scaledz_df = results[scaled_z_scores]
            for col in scaledz_df.columns:
                explain_df[col] = scaledz_df[col]

            zscores = (rloss_scores - batch.train_scores_mean) / batch.train_scores_std
            rloss_scores = rloss_scores.reshape((batch.count, 1))
            zscores = np.absolute(zscores)
            zscores = zscores.reshape((batch.count, 1))
        else:
            rloss_scores = np.empty((batch.count, 1))
            rloss_scores[:] = np.NaN
            zscores = np.empty((batch.count, 1))
            zscores[:] = np.NaN

        ae_scores = np.concatenate((rloss_scores, zscores), axis=1)

        ae_scores = cp.asarray(ae_scores)

        mem = ResponseMemoryAE(count=batch.count, probs=ae_scores)

        mem.explain_df = explain_df

        cb(mem)


@register_stage("inf-pytorch", modes=[PipelineModes.AE])
class AutoEncoderInferenceStage(InferenceStage):
    """
    Perform inference with PyTorch.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._config = c

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

        return _AutoEncoderInferenceWorker(inf_queue, self._config)

    @staticmethod
    def _convert_one_response(output: MultiResponseProbsMessage, inf: MultiInferenceMessage, res: ResponseMemoryAE):

        # Set the explainability and then call the base
        res.explain_df.index = range(inf.mess_offset, inf.mess_offset + inf.mess_count)
        for col in res.explain_df.columns:
            inf.set_meta(col, res.explain_df[col])

        return InferenceStage._convert_one_response(output=output, inf=inf, res=res)
