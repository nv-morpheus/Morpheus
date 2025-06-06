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

import mrc
from gliner import GLiNER
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_aliases import DataFrameType


@register_stage("gliner-processor")
class GliNERProcessor(GpuAndCpuMixin, ControlMessageStage):
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    labels : list[str]
        List of entity labels to detect, this should match the named patterns used in the RegexProcessor stage.
    model_name : str
        Name of the model to use.
    column_name : str
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
                 labels: list[str],
                 model_name: str = "gretelai/gretel-gliner-bi-small-v1.0",
                 column_name: str = "source_text",
                 confidence_threshold: float = 0.7,
                 context_window: int = 100,
                 fallback: bool = True):

        super().__init__(config)
        self.confidence_threshold = confidence_threshold
        self.entity_labels = labels

        # Load the fine-tuned GLiNER model
        if config.execution_mode == ExecutionMode.GPU:
            map_location = "cuda"
        else:
            map_location = "cpu"

        self.model = GLiNER.from_pretrained(model_name, map_location=map_location)
        self.column_name = column_name
        self.context_window = context_window
        self.fallback = fallback
        self._needed_columns['dlp_findings'] = TypeId.STRING
        self._model_max_batch_size = config.model_max_batch_size

    @property
    def name(self) -> str:
        return "gliner-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def _extract_contexts_from_regex_findings(
            self, text: str, regex_findings: list[dict[str, typing.Any]]) -> tuple[list[str], list[tuple[int, int]]]:
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

        for finding in regex_findings:
            span = finding.get("span")
            if span:
                start, end = span

                # Expand the context window with single min/max calls
                context_start = max(0, start - context_window)
                context_end = min(text_len, end + context_window)

                # Only add if this span is unique
                span_key = (context_start, context_end)
                if span_key not in unique_spans:
                    unique_spans.add(span_key)
                    contexts.append(text[context_start:context_end])
                    spans.append(span_key)

        # If no valid contexts were extracted, use the full text
        if not contexts:
            contexts.append(text)
            spans.append((0, len(text)))

        return (contexts, spans)

    def _prepare_data(self, rows: list[dict[str, typing.Any]]) -> tuple[list[str], list[tuple[int, int]], list[int]]:
        """
        Prepare the data for processing by ensuring the necessary columns are present.
        """
        model_data = []
        model_row_to_row_num = []
        all_spans = []
        for (i, row) in enumerate(rows):
            regex_findings = row['regex_findings']
            text = row[self.column_name]
            if regex_findings is not None and len(regex_findings) > 0:

                contexts, spans = self._extract_contexts_from_regex_findings(text, regex_findings)
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

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        with msg.payload().mutable_dataframe() as df:
            dlp_findings = []
            rows = df[[self.column_name, 'regex_findings']].to_dict(orient="records")

            (model_data, all_spans, model_row_to_row_num) = self._prepare_data(rows)

            model_entities = []
            for i in range(0, len(model_data), self._model_max_batch_size):
                batch_data = model_data[i:i + self._model_max_batch_size]

                model_entities.extend(
                    self.model.batch_predict_entities(batch_data,
                                                      self.entity_labels,
                                                      flat_ner=True,
                                                      threshold=self.confidence_threshold,
                                                      multi_label=False))

            dlp_findings = self._process_results(len(rows), model_entities, all_spans, model_row_to_row_num)

            df['dlp_findings'] = dlp_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
