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

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_aliases import DataFrameType


@register_stage("regex-processor")
class RegexProcessor(ControlMessageStage, GpuAndCpuMixin):
    """Process text with regex patterns to identify structured sensitive data"""

    def __init__(self, patterns: dict[str, list[str]], column_name: str = "source_text"):
        """
        Initialize with regex patterns to detect sensitive data

        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        self.column_name = column_name
        self.combined_patterns = {}

        # For each entity type, combine multiple patterns into a single regex
        for entity_type, pattern_list in patterns.items():

            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
            else:
                combined_pattern = pattern_list[0]

            self.combined_patterns[entity_type] = combined_pattern

    @property
    def name(self) -> str:
        return "regex-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Scan text for sensitive data using regex patterns

        Returns:
            List of findings with metadata
        """

        with msg.payload().mutable_dataframe() as df:
            # Extract the text column to process
            text_series = df[self.column_name]
            for pattern_name, pattern in self.combined_patterns.items():
                # for pattern in pattern_list:
                match_df: DataFrameType = text_series.str.extract(pattern, expand=True)
                for column in match_df:
                    df[f"regex_{pattern_name}_{column}"] = match_df[column]

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
