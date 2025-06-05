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
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin


@register_stage("gliner-processor")
class GliNERProcessor(ControlMessageStage, GpuAndCpuMixin):
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text
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
        """
        Initialize with configuration for SLM-based detection

        Args:
            confidence_threshold: Minimum confidence score to report a finding
        """
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

    @property
    def name(self) -> str:
        return "gliner-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def gliner_predict(self, text: str) -> list[dict[str, list]]:
        """
        Predict entities in text using GLiNER
        """
        results = self.model.predict_entities(text,
                                              self.entity_labels,
                                              flat_ner=True,
                                              threshold=self.confidence_threshold,
                                              multi_label=False)

        return results

    def filter_entities(self, entities: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        Filter entities for relevant keys
        """
        entities = [{'label': r['label'], 'start': r['start'], 'end': r['end'], 'score': r['score']} for r in entities]
        return entities

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

    def _process_row(self, text: str,
                     regex_findings: list[dict[str, typing.Any]] | None) -> list[dict[str, typing.Any]]:
        all_entities = []
        if regex_findings is not None and len(regex_findings) > 0:

            contexts, spans = self._extract_contexts_from_regex_findings(text, regex_findings)
            assert len(contexts) == len(spans)
            model_entities = self.model.batch_predict_entities(contexts,
                                                               self.entity_labels,
                                                               flat_ner=True,
                                                               threshold=self.confidence_threshold,
                                                               multi_label=False)

            seen = set()
            unique_entities = []
            for i, entities in enumerate(model_entities):
                span_offset = spans[i][0]
                for entity in entities:
                    entity["start"] += span_offset
                    entity["end"] += span_offset
                    entity_key = (entity["label"], entity["text"], entity["start"], entity["end"])
                    if entity_key not in seen:
                        seen.add(entity_key)
                        unique_entities.append(entity)

            all_entities = unique_entities
        elif self.fallback:
            model_entities = self.gliner_predict(text)
            all_entities = self.filter_entities(model_entities)

        return all_entities

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection

        Args:
            text: The text to analyze
            regex_findings: Optional list of regex findings to filter candidates for classification

        Returns:
            List of findings with metadata
        """

        with msg.payload().mutable_dataframe() as df:
            dlp_findings = []
            rows = df[[self.column_name, 'regex_findings']].to_dict(orient="records")

            for row in rows:
                regex_findings = row['regex_findings']
                text = row[self.column_name]

                dlp_findings.append(self._process_row(text, regex_findings))

            df['dlp_findings'] = dlp_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
