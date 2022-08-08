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

import cupy as cp
from morpheus.cli.register_stage import register_stage

from morpheus.config import Config, PipelineModes
from morpheus.messages import MultiResponseAEMessage
from morpheus.messages import ResponseMemory
from morpheus.messages import ResponseMemoryProbs
from morpheus.messages.multi_inference_ae_message import MultiInferenceAEMessage
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

        memory = ResponseMemoryProbs(count=output_dims[0], probs=cp.zeros(output_dims))

        # Override the default to return the response AE
        output_message = MultiResponseAEMessage(meta=x.meta,
                                                mess_offset=x.mess_offset,
                                                mess_count=x.mess_count,
                                                memory=memory,
                                                offset=0,
                                                count=memory.count,
                                                user_id=x.user_id)
        return output_message

    def calc_output_dims(self, x: MultiInferenceAEMessage) -> typing.Tuple:

        # We only want one score
        return (x.count, 1)

    def process(self, batch: MultiInferenceAEMessage, cb: typing.Callable[[ResponseMemory], None]):
        """
        This function processes inference batch by using batch's model to calculate anomaly scores
        and adding results to response.

        Parameters
        ----------
        batch : `morpheus.pipeline.messagesMultiInferenceMessage`
            Batch of inference messages.
        cb : typing.Callable[[`morpheus.pipeline.messages.ResponseMemory`], None]
            Inference callback.

        """
        data = batch.get_meta(batch.meta.df.columns.intersection(self._feature_columns))

        net_loss = batch.model.get_anomaly_score(data)
        ae_scores = cp.asarray(net_loss)
        ae_scores = ae_scores.reshape((batch.count, 1))

        mem = ResponseMemoryProbs(
            count=batch.count,
            probs=ae_scores,  # For now, only support one output
        )

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
