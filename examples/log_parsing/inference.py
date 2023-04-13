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

import logging
import typing
from functools import partial

import cupy as cp
import mrc
import numpy as np
import tritonclient.grpc as tritonclient
from mrc.core import operators as ops
from scipy.special import softmax

from messages import MultiPostprocLogParsingMessage
from messages import MultiResponseLogParsingMessage
from messages import PostprocMemoryLogParsing
from messages import ResponseMemoryLogParsing
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemory
from morpheus.messages import MultiInferenceMessage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.inference.inference_stage import InferenceStage
from morpheus.stages.inference.inference_stage import InferenceWorker
from morpheus.stages.inference.triton_inference_stage import InputWrapper
from morpheus.stages.inference.triton_inference_stage import _TritonInferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue

logger = logging.getLogger(__name__)


class TritonInferenceLogParsing(_TritonInferenceWorker):
    """
    This class extends TritonInference to deal with scenario-specific NLP models inference requests like building
    response.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton
        inference server.
    server_url : str
        Triton server gRPC URL including the port.
    inout_mapping : typing.Dict[str, str]
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model
    use_shared_memory: bool, default = True
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine

    """

    def __init__(self,
                 inf_queue: ProducerConsumerQueue,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool,
                 use_shared_memory: bool,
                 inout_mapping: typing.Dict[str, str] = None):
        # Some models use different names for the same thing. Set that here but allow user customization
        default_mapping = {
            "attention_mask": "input_mask",
        }

        default_mapping.update(inout_mapping if inout_mapping is not None else {})

        super().__init__(inf_queue,
                         c,
                         model_name=model_name,
                         server_url=server_url,
                         force_convert_inputs=force_convert_inputs,
                         use_shared_memory=use_shared_memory,
                         inout_mapping=default_mapping)

    @classmethod
    def needs_logits(cls):
        return True

    @classmethod
    def default_inout_mapping(cls) -> typing.Dict[str, str]:
        # Some models use different names for the same thing. Set that here but allow user customization
        return {"attention_mask": "input_mask"}

    def build_output_message(self, x: MultiInferenceMessage) -> MultiResponseLogParsingMessage:

        memory = PostprocMemoryLogParsing(
            count=x.count,
            confidences=cp.zeros((x.count, self._inputs[list(self._inputs.keys())[0]].shape[1])),
            labels=cp.zeros((x.count, self._inputs[list(self._inputs.keys())[0]].shape[1])),
            input_ids=cp.zeros((x.count, x.input_ids.shape[1])),
            seq_ids=cp.zeros((x.count, x.seq_ids.shape[1])),
        )

        output_message = MultiPostprocLogParsingMessage(meta=x.meta,
                                                        mess_offset=x.mess_offset,
                                                        mess_count=x.mess_count,
                                                        memory=memory,
                                                        offset=x.offset,
                                                        count=x.count)
        return output_message

    def _build_response(self, batch: MultiInferenceMessage,
                        result: tritonclient.InferResult) -> ResponseMemoryLogParsing:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}
        output = {key: softmax(val, axis=2) for key, val in output.items()}
        confidences = {key: np.amax(val, axis=2) for key, val in output.items()}
        labels = {key: np.argmax(val, axis=2) for key, val in output.items()}

        mem = ResponseMemoryLogParsing(
            count=output[list(output.keys())[0]].shape[0],
            confidences=cp.array(confidences[list(output.keys())[0]]),
            labels=cp.array(labels[list(output.keys())[0]]),
        )

        return mem

    def _infer_callback(self,
                        cb: typing.Callable[[ResponseMemoryLogParsing], None],
                        m: InputWrapper,
                        b: MultiInferenceMessage,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException):

        # If its an error, return that here
        if (error is not None):
            raise error

        # Build response
        response_mem = self._build_response(b, result)

        # Call the callback with the memory
        cb(response_mem)

        self._mem_pool.return_obj(m)


@register_stage("inf-logparsing", modes=[PipelineModes.NLP])
class LogParsingInferenceStage(InferenceStage):
    """
    NLP Triton inference stage for log parsing pipeline.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    model_name : str
        Name of the model specifies which model can handle the inference requests that are sent to Triton inference
        server.
    server_url : str
        Triton server URL
    force_convert_inputs : bool, default = False
        Instructs the stage to convert the incoming data to the same format that Triton is expecting. If set to False,
        data will only be converted if it would not result in the loss of data.
    use_shared_memory: bool, default = False, is_flag = True
        Whether or not to use CUDA Shared IPC Memory for transferring data to Triton. Using CUDA IPC reduces network
        transfer time but requires that Morpheus and Triton are located on the same machine

    """

    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool = False,
                 use_shared_memory: bool = False):
        super().__init__(c)

        self._config = c

        self._kwargs = {
            "model_name": model_name,
            "server_url": server_url,
            "force_convert_inputs": force_convert_inputs,
            "use_shared_memory": use_shared_memory,
        }

        self._requires_seg_ids = False

    def supports_cpp_node(self):
        # Get the value from the worker class
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiResponseLogParsingMessage

        def py_inference_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            worker = self._get_inference_worker(self._inf_queue)

            worker.init()

            outstanding_requests = 0

            def on_next(x: MultiInferenceMessage):
                nonlocal outstanding_requests

                batches = self._split_batches(x, self._max_batch_size)

                output_message = worker.build_output_message(x)

                memory = output_message.memory

                fut_list = []

                for batch in batches:
                    outstanding_requests += 1

                    fut = mrc.Future()

                    def set_output_fut(resp: ResponseMemoryLogParsing, b, f: mrc.Future):
                        nonlocal outstanding_requests
                        m = self._convert_one_response(memory, b, resp)

                        f.set_result(m)

                        outstanding_requests -= 1

                    fut_list.append(fut)

                    worker.process(batch, partial(set_output_fut, b=batch, f=fut))

                for f in fut_list:
                    f.result()

                return output_message

            obs.pipe(ops.map(on_next)).subscribe(sub)

            assert outstanding_requests == 0, "Not all inference requests were completed"

        if (self._build_cpp_node()):
            node = self._get_cpp_inference_node(builder)
        else:
            node = builder.make_node(self.unique_name, ops.build(py_inference_fn))

        # Set the concurrency level to be up with the thread count
        node.launch_options.pe_count = self._thread_count
        builder.make_edge(stream, node)

        stream = node

        return stream, out_type

    @staticmethod
    def _convert_one_response(memory: InferenceMemory, inf: MultiInferenceMessage, res: ResponseMemoryLogParsing):

        memory.input_ids[inf.offset:inf.count + inf.offset, :] = inf.input_ids
        memory.seq_ids[inf.offset:inf.count + inf.offset, :] = inf.seq_ids

        # Two scenarios:
        if (inf.mess_count == inf.count):
            memory.confidences[inf.offset:inf.count + inf.offset, :] = res.confidences
            memory.labels[inf.offset:inf.count + inf.offset, :] = res.labels
        else:
            assert inf.count == res.count

            mess_ids = inf.seq_ids[:, 0].get().tolist()

            # Out message has more reponses, so we have to do key based blending of probs
            for i, idx in enumerate(mess_ids):
                memory.confidences[idx, :] = cp.maximum(memory.confidences[idx, :], res.confidences[i, :])
                memory.labels[idx, :] = cp.maximum(memory.labels[idx, :], res.labels[i, :])

        return MultiPostprocLogParsingMessage(meta=inf.meta,
                                              mess_offset=inf.mess_offset,
                                              mess_count=inf.mess_count,
                                              memory=memory,
                                              offset=inf.offset,
                                              count=inf.count)

    def _get_inference_worker(self, inf_queue: ProducerConsumerQueue) -> InferenceWorker:

        return TritonInferenceLogParsing(inf_queue, self._config, **self._kwargs)
