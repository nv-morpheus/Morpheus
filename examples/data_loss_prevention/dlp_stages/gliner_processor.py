# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from functools import partial

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin

from .gliner_triton import GliNERTritonInference

EntitiesType = list[dict[str, typing.Any]]


@register_stage("gliner-processor")
class GliNERProcessor(GpuAndCpuMixin, ControlMessageStage):
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    model_source_dir : str
        Path to the directory containing the GLiNER model files. Used for pre and post-processing.
    labels : list[str]
        List of entity labels to detect, this should match the named patterns used in the RegexProcessor stage.
    onnx_path : str, default = "model.onnx"
        Path to the ONNX model file, relative to the `model_source_dir`. Default is "model.onnx".
    server_url : str, default = "localhost:8001"
        URL of the Triton inference server.
    triton_model_name : str, default = "gliner-bi-encoder-onnx"
        Name of the Triton model to use for inference.
    source_column_name : str, default = "source_text"
        Name of the column containing the source text to process.
    confidence_threshold: float, default = 0.3
        Minimum confidence score to report a finding
    context_window : int, default = 100
        Number of characters before and after a regex match to include in the context for SLM analysis.
    fallback : bool, default = False
        If True, fallback to GLiNER prediction if no regex findings are available.
        If False, only process rows with regex findings.
    fallback_model_name: str, default="gretelai/gretel-gliner-bi-small-v1.0"
        Name of the fallback model to use if the local model is not present.
    """

    def __init__(self,
                 config: Config,
                 *,
                 model_source_dir: str,
                 labels: list[str],
                 onnx_path: str = "model.onnx",
                 server_url: str = "localhost:8001",
                 triton_model_name: str = "gliner-bi-encoder-onnx",
                 source_column_name: str = "source_text",
                 confidence_threshold: float = 0.3,
                 context_window: int = 100,
                 fallback: bool = False,
                 fallback_model_name: str = "gretelai/gretel-gliner-bi-small-v1.0"):

        super().__init__(config)
        if config.execution_mode == ExecutionMode.GPU:
            map_location = "cuda"
        else:
            map_location = "cpu"

        self._model_max_batch_size = config.model_max_batch_size
        self.source_column_name = source_column_name
        self._confidence_threshold = confidence_threshold
        self.context_window = context_window
        self.fallback = fallback
        self.gliner_triton = GliNERTritonInference(server_url=server_url,
                                                   triton_model_name=triton_model_name,
                                                   model_source_dir=model_source_dir,
                                                   onnx_path=onnx_path,
                                                   map_location=map_location,
                                                   labels=labels,
                                                   gliner_threshold=confidence_threshold,
                                                   fallback_model_name=fallback_model_name)

    @property
    def name(self) -> str:
        return "gliner-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def _process_results(self, batch_entities: list[list[EntitiesType]]) -> list[EntitiesType]:
        dlp_findings = []

        # flattend the batch_entities list, currently each entry in the batch_entities represents a batch of entities
        # by flattening it we get a list of entities for each row in the input DataFrame
        for entities in batch_entities:
            dlp_findings.extend(entities)

        return dlp_findings

    def _infer_callback(self,
                        *,
                        batch_num: int,
                        batch_entities: list[list[EntitiesType]],
                        future: mrc.Future,
                        entities: list[EntitiesType]):
        batch_entities[batch_num] = entities
        future.set_result(batch_num)

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        with msg.payload().mutable_dataframe() as df:
            input_data = df[self.source_column_name]
            if not isinstance(input_data, pd.Series):
                input_data = input_data.to_arrow().to_pylist()
            else:
                input_data = input_data.tolist()

            futures = []
            batch_entities = []
            for i in range(0, len(input_data), self._model_max_batch_size):
                future = mrc.Future()
                futures.append(future)
                batch_entities.append(None)
                batch_data = input_data[i:i + self._model_max_batch_size]

                self.gliner_triton.process(
                    batch_data,
                    partial(self._infer_callback,
                            batch_num=len(batch_entities) - 1,
                            batch_entities=batch_entities,
                            future=future))

            for future in futures:
                future.result()

            dlp_findings = self._process_results(batch_entities)

            df['dlp_findings'] = dlp_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
