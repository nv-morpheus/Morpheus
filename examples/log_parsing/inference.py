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

import cupy as cp
import numpy as np
import tritonclient.grpc as tritonclient
from scipy.special import softmax

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemory
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ResponseMemory
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceWorker

logger = logging.getLogger(__name__)


class TritonInferenceLogParsing(TritonInferenceWorker):
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
    needs_logits : bool, default = True
        Determines whether a logits calculation is needed for the value returned by the Triton inference response.
    """

    def build_output_message(self, x: MultiInferenceMessage) -> MultiResponseMessage:
        seq_ids = cp.zeros((x.count, 3), dtype=cp.uint32)
        seq_ids[:, 0] = cp.arange(x.mess_offset, x.mess_offset + x.count, dtype=cp.uint32)
        seq_ids[:, 2] = x.seq_ids[:, 2]

        memory = ResponseMemory(
            count=x.count,
            tensors={
                'confidences': cp.zeros((x.count, self._inputs[list(self._inputs.keys())[0]].shape[1])),
                'labels': cp.zeros((x.count, self._inputs[list(self._inputs.keys())[0]].shape[1])),
                'input_ids': cp.zeros((x.count, x.input_ids.shape[1])),
                'seq_ids': seq_ids
            })

        return MultiResponseMessage(meta=x.meta,
                                    mess_offset=x.mess_offset,
                                    mess_count=x.mess_count,
                                    memory=memory,
                                    offset=0,
                                    count=x.count)

    def _build_response(self, batch: MultiInferenceMessage, result: tritonclient.InferResult) -> ResponseMemory:

        outputs = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}
        outputs = {key: softmax(val, axis=2) for key, val in outputs.items()}
        confidences = {key: np.amax(val, axis=2) for key, val in outputs.items()}
        labels = {key: np.argmax(val, axis=2) for key, val in outputs.items()}

        return ResponseMemory(count=outputs[list(outputs.keys())[0]].shape[0],
                              tensors={
                                  'confidences': cp.array(confidences[list(outputs.keys())[0]]),
                                  'labels': cp.array(labels[list(outputs.keys())[0]])
                              })


@register_stage("inf-logparsing", modes=[PipelineModes.NLP])
class LogParsingInferenceStage(TritonInferenceStage):
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
    needs_logits : bool, default = True, is_flag = True
        Determines whether a logits calculation is needed for the value returned by the Triton inference response.
    inout_mapping : dict[str, str], optional
        Dictionary used to map pipeline input/output names to Triton input/output names. Use this if the
        Morpheus names do not match the model.
    """

    def __init__(self,
                 c: Config,
                 model_name: str,
                 server_url: str,
                 force_convert_inputs: bool = False,
                 use_shared_memory: bool = False,
                 needs_logits: bool = True,
                 inout_mapping: dict[str, str] = None):
        super().__init__(c,
                         model_name=model_name,
                         server_url=server_url,
                         force_convert_inputs=force_convert_inputs,
                         use_shared_memory=use_shared_memory,
                         needs_logits=needs_logits,
                         inout_mapping=inout_mapping)

    def supports_cpp_node(self) -> bool:
        # Get the value from the worker class
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MultiResponseMessage)

    @staticmethod
    def _convert_one_response(output: MultiResponseMessage, inf: MultiInferenceMessage,
                              res: ResponseMemory) -> MultiResponseMessage:

        output.input_ids[inf.offset:inf.count + inf.offset, :] = inf.input_ids
        output.seq_ids[inf.offset:inf.count + inf.offset, :] = inf.seq_ids

        # Two scenarios:
        if (inf.mess_count == inf.count):
            output.confidences[inf.offset:inf.count + inf.offset, :] = res.get_output('confidences')
            output.labels[inf.offset:inf.count + inf.offset, :] = res.get_output('labels')
        else:
            assert inf.count == res.count

            mess_ids = inf.seq_ids[:, 0].get().tolist()

            # Out message has more reponses, so we have to do key based blending of probs
            for i, idx in enumerate(mess_ids):
                output.confidences[idx, :] = cp.maximum(output.confidences[idx, :], res.get_output('confidences')[i, :])
                output.labels[idx, :] = cp.maximum(output.labels[idx, :], res.get_output('labels')[i, :])

        return MultiResponseMessage.from_message(inf, memory=output.memory, offset=inf.offset, count=inf.mess_count)

    def _get_worker_class(self) -> type[TritonInferenceWorker]:
        return TritonInferenceLogParsing
