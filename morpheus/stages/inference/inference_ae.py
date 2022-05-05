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

import dataclasses
import typing

import cupy as cp
from dfencoder.autoencoder import AutoEncoder

from morpheus.config import Config
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseAEMessage
from morpheus.messages import ResponseMemory
from morpheus.messages import ResponseMemoryProbs
from morpheus.messages import UserMessageMeta
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


@dataclasses.dataclass
class MultiInferenceAEMessage(MultiInferenceMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for AE workloads. Helps ensure the
    proper inputs are set and eases debugging. Associates a user ID with a message.
    """

    model: AutoEncoder

    @property
    def user_id(self):
        """
        Returns the user ID associated with this message.

        """

        return typing.cast(UserMessageMeta, self.meta).user_id

    @property
    def input(self):
        """
        Returns autoecoder input tensor.

        Returns
        -------
        cupy.ndarray
            The autoencoder input tensor.

        """

        return self.get_input("input")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            seq_ids

        """

        return self.get_input("seq_ids")

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MultiInferenceAEMessage`
            A new `MultiInferenceAEMessage` with sliced offset and count.

        """
        mess_start = self.mess_offset + self.seq_ids[start, 0].item()
        mess_stop = self.mess_offset + self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceAEMessage(meta=self.meta,
                                       mess_offset=mess_start,
                                       mess_count=mess_stop - mess_start,
                                       memory=self.memory,
                                       offset=start,
                                       count=stop - start,
                                       model=self.model)


class AutoEncoderInference(InferenceWorker):

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

        output_dims = self.calc_output_dims(x)

        memory = ResponseMemoryProbs(count=x.count, probs=cp.zeros(output_dims))

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


class AutoEncoderInferenceStage(InferenceStage):
    """
    Inference stage for the AE pipeline.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._config = c

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

        return AutoEncoderInference(inf_queue, self._config)
