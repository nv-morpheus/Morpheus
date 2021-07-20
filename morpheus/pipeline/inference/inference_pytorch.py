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

import asyncio
import queue
import typing

import cupy as cp
from tornado.ioloop import IOLoop

from morpheus.config import Config
from morpheus.pipeline.inference.inference_stage import InferenceStage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import ResponseMemoryProbs

try:
    import torch
    from torch.utils.dlpack import from_dlpack
    from torch.utils.dlpack import to_dlpack
except ImportError:
    print(("PyTorch Not Found! PyTorch must be installed to use the PyTorchInferenceStage. "
           "Due to the limited CUDA options available in the PyTorch stable versions, "
           "it must be manually installed by the user. "
           "Please see the Getting Started Guide: https://pytorch.org/get-started/locally/"))
    raise


class PyTorchInference:
    def __init__(self, c: Config, model_filename: str):

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.feature_length
        self._model_filename: str = model_filename

        self._loop: IOLoop = None
        self._model = None

    def init(self, loop: IOLoop):

        self._loop = loop

        # Load the model into CUDA memory
        self._model = torch.load(self._model_filename).to('cuda')

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(batch.get_input("input_ids").astype(cp.float).toDlpack()).type(torch.long)
        attention_mask = from_dlpack(batch.get_input("input_mask").astype(cp.float).toDlpack()).type(torch.long)

        with torch.no_grad():
            logits = self._model(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
            probs = torch.sigmoid(logits)

        probs_cp = cp.fromDlpack(to_dlpack(probs))

        # Ensure that we are of the shape `[Batch Size, Num Labels]`
        if (len(probs_cp.shape) == 1):
            probs_cp = cp.expand_dims(probs_cp, axis=1)

        response_mem = ResponseMemoryProbs(count=batch.count, probs=probs_cp)

        def tmp(mem: ResponseMemoryProbs):

            # Set result on future
            fut.set_result(mem)

        self._loop.add_callback(tmp, response_mem)

    def main_loop(self, loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

        self.init(loop)

        if (ready_event is not None):
            loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            self.process(batch, fut)


class PyTorchInferenceStage(InferenceStage):
    def __init__(self, c: Config, model_filename: str):
        super().__init__(c)

        self._model_filename = model_filename

    def _get_inference_fn(self) -> typing.Callable:

        worker = PyTorchInference(Config.get(), model_filename=self._model_filename)

        return worker.main_loop
