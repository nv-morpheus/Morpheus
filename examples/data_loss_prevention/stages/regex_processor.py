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

import json
import logging
import pathlib
import re

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("regex-processor")
class RegexProcessor(ControlMessageStage, GpuAndCpuMixin):
    """Process text with regex patterns to identify structured sensitive data"""

    def __init__(self,
                 config: Config,
                 *,
                 patterns: dict[str, list[str]] | None = None,
                 patterns_file: str | pathlib.Path | None = None,
                 column_name: str = "source_text",
                 confidence: float = 0.9):
        """
        Initialize with regex patterns to detect sensitive data

        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        super().__init__(config)
        self.column_name = column_name
        self.combined_patterns = {}
        self.confidence = confidence

        if patterns is None:
            if patterns_file is None:
                raise ValueError("Either 'patterns' or 'patterns_file' must be provided")
            patterns = self.load_regex_patterns(patterns_file)
            logger.info("Loaded %d regex pattern groups", len(patterns))

        # For each entity type, combine multiple patterns into a single regex
        for entity_type, pattern_list in patterns.items():

            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
            else:
                combined_pattern = pattern_list[0]

            self.combined_patterns[entity_type] = re.compile(combined_pattern)

    @staticmethod
    def load_regex_patterns(file_path: str | pathlib.Path) -> dict[str, list[str]]:
        """Load regex patterns from a JSON file."""
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    @property
    def name(self) -> str:
        return "regex-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @property
    def patterns(self) -> dict[str, re.Pattern]:
        """
        Returns the compiled regex patterns used for detection.
        """
        return self.combined_patterns.copy()

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Scan text for sensitive data using regex patterns

        Returns:
            List of findings with metadata
        """

        with msg.payload().mutable_dataframe() as df:
            # Extract the text column to process
            text_series = df[self.column_name]
            if not isinstance(text_series, pd.Series):
                # cudf series doesn't support iteration
                text_series = text_series.to_arrow().to_pylist()

            all_findings = []
            for text in text_series:
                findings = []
                for (pattern_name, pattern) in self.combined_patterns.items():
                    matches = pattern.finditer(text)
                    for match in matches:
                        findings.append({
                            "label": pattern_name,
                            "match": match.group(),
                            "span": match.span(),
                            "detection_method": "regex",
                            "confidence": self.confidence
                        })

                all_findings.append(findings)

            df['regex_findings'] = all_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
