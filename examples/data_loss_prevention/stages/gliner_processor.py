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

import logging
import typing
from functools import partial

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin

from .gliner_triton import GliNERTritonInference

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("gliner-processor")
class GliNERProcessor(GpuAndCpuMixin, ControlMessageStage):
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    model_name : str
        Name of the model to use.
    source_column_name : str
        Name of the column containing the source text to process.
    context_window : int
        Number of characters before and after a regex match to include in the context for SLM analysis.
    confidence_threshold: float
        Minimum confidence score to report a finding
    fallback : bool
        If True, fallback to GLiNER prediction if no regex findings are available.
        If False, only process rows with regex findings.
    """

    def __init__(self,
                 config: Config,
                 *,
                 model_source_dir: str | None = None,
                 model_name: str = "gretelai/gretel-gliner-bi-small-v1.0",
                 source_column_name: str = "source_text",
                 regex_col_prefix: str = "regex_matches_",
                 confidence_threshold: float = 0.3,
                 context_window: int = 100,
                 fallback: bool = True):

        super().__init__(config)
        if config.execution_mode == ExecutionMode.GPU:
            map_location = "cuda"
        else:
            map_location = "cpu"

        self._model_name = model_name
        self._model_max_batch_size = config.model_max_batch_size
        self.source_column_name = source_column_name
        self._regex_col_prefix = regex_col_prefix
        self._confidence_threshold = confidence_threshold
        self.context_window = context_window
        self.fallback = fallback
        self._needed_columns['dlp_findings'] = TypeId.STRING
        self.gliner_triton = GliNERTritonInference(model_source_dir=model_source_dir,
                                                   map_location=map_location,
                                                   gliner_threshold=confidence_threshold)

    @property
    def name(self) -> str:
        return "gliner-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def _extract_contexts_from_regex_findings(
            self, text: str, row: dict[str, typing.Any],
            regex_columns: list[str]) -> tuple[list[str], list[str], list[tuple[int, int]]]:
        """
        Extract text contexts around regex matches to focus SLM analysis

        Args:
            text: The full text being analyzed
            regex_findings: List of regex findings with span information

        Returns:
            List of text contexts for focused analysis
        """
        contexts = []
        spans = []
        context_window = self.context_window  # Characters before and after the match

        # Track unique spans to avoid duplicates
        # Pre-allocate lists and use set for O(1) lookups
        unique_spans = set()
        text_len = len(text)

        regex_findings = []
        for regex_col in regex_columns:
            findings = row[regex_col]
            if isinstance(findings, list) and len(findings) > 0:
                regex_findings.extend(findings)

            for finding in findings:
                start = text.find(finding)

                if start > -1:  # Ensure the finding was found in the text
                    end = start + len(finding)

                    # Expand the context window with single min/max calls
                    context_start = max(0, start - context_window)
                    context_end = min(text_len, end + context_window)

                    # Only add if this span is unique
                    span_key = (context_start, context_end)
                    if span_key not in unique_spans:
                        unique_spans.add(span_key)
                        contexts.append(text[context_start:context_end])
                        spans.append(span_key)
                else:
                    logger.warning("Regex finding '%s' not found in text: %s", finding, text)

        # If no valid contexts were extracted, use the full text
        if not contexts:
            contexts.append(text)
            spans.append((0, len(text)))

        return (regex_findings, contexts, spans)

    def _prepare_data(self, rows: list[dict[str, typing.Any]],
                      regex_columns: list[str]) -> tuple[list[str], list[tuple[int, int]], list[int]]:
        """
        Prepare the data for processing by ensuring the necessary columns are present.
        """
        model_data = []
        model_row_to_row_num = []
        all_spans = []
        for (i, row) in enumerate(rows):

            text = row[self.source_column_name]
            (regex_findings, contexts, spans) = self._extract_contexts_from_regex_findings(text, row, regex_columns)
            if len(regex_findings) > 0:
                assert len(contexts) == len(spans)
                model_data.extend(contexts)
                all_spans.append(spans)
                model_row_to_row_num.extend([i] * len(contexts))
            elif self.fallback:
                # If fallback is enabled, process the full text
                model_data.append(text)
                all_spans.append([(0, len(text))])
                model_row_to_row_num.append(i)

        assert len(model_data) == len(model_row_to_row_num), "Mismatch between contexts and row numbers"
        return (model_data, all_spans, model_row_to_row_num)

    def _process_one_result(self, model_entities: list[dict[str, typing.Any]],
                            spans: list[tuple[int, int]]) -> list[dict[str, typing.Any]]:
        seen = set()
        unique_entities = []
        for k, entities in enumerate(model_entities):
            span_offset = spans[k][0]
            for entity in entities:
                entity["start"] += span_offset
                entity["end"] += span_offset
                entity_key = (entity["label"], entity["text"], entity["start"], entity["end"])
                if entity_key not in seen:
                    seen.add(entity_key)
                    unique_entities.append(entity)

        return unique_entities

    def _process_results(self,
                         num_rows: int,
                         model_entities: list[list[dict[str, typing.Any]]],
                         all_spans: list[tuple[int, int]],
                         model_row_to_row_num: list[int]) -> list[list[dict[str, typing.Any]]]:
        dlp_findings = [[] for _ in range(num_rows)]

        # flattend the model_entities list
        _flat = []
        for entities in model_entities:
            assert entities is not None
            _flat.extend(entities)

        model_entities = _flat

        entities_per_row = None
        current_row = None
        for (i, entities) in enumerate(model_entities):
            row_num = model_row_to_row_num[i]
            if row_num != current_row:
                if current_row is not None:
                    spans = all_spans[current_row]
                    dlp_findings[current_row] = self._process_one_result(entities_per_row, spans)

                current_row = row_num
                entities_per_row = []

            entities_per_row.append(entities)

        # Process the last row
        if current_row is not None:
            spans = all_spans[current_row]
            dlp_findings[current_row] = self._process_one_result(entities_per_row, spans)
        return dlp_findings

    def _infer_callback(self,
                        *,
                        batch_num: int,
                        model_entities: list[list[dict[str, typing.Any]]],
                        future: mrc.Future,
                        entities: list[list[dict[str, typing.Any]]]):
        model_entities[batch_num] = entities
        future.set_result(batch_num)

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        with msg.payload().mutable_dataframe() as df:
            dlp_findings = []

            regex_columns = [col for col in df.columns if col.startswith(self._regex_col_prefix)]
            rows = df[[self.source_column_name] + regex_columns].to_dict(orient="records")

            (model_data, all_spans, model_row_to_row_num) = self._prepare_data(rows, regex_columns)

            batches = []
            model_entities = []
            for i in range(0, len(model_data), self._model_max_batch_size):
                batch_data = model_data[i:i + self._model_max_batch_size]
                batches.append(batch_data)
                model_entities.append(None)

            futures = []
            for (batch_num, batch) in enumerate(batches):
                future = mrc.Future()
                futures.append(future)
                self.gliner_triton.process(
                    batch,
                    partial(self._infer_callback, batch_num=batch_num, model_entities=model_entities, future=future))

            for future in futures:
                future.result()

            dlp_findings = self._process_results(len(rows), model_entities, all_spans, model_row_to_row_num)

            df['dlp_findings'] = dlp_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
