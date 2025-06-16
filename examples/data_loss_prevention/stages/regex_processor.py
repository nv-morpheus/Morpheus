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

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("regex-processor")
class RegexProcessor(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Process text with regex patterns to identify structured sensitive data

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    patterns: dict[str, list[str]] | None
        Dictionary mapping data types to lists of regex patterns
    patterns_file : str | pathlib.Path | None
        Path to a JSON file containing regex patterns for different data types.
        Ignored if `patterns` is provided.
    source_column_name : str
        Name of the column containing the source text to process.
    include_pattern_names : bool
        If True, the output will include the names of the patterns that matched.
    """

    def __init__(self,
                 config: Config,
                 *,
                 patterns: dict[str, list[str]] | None = None,
                 patterns_file: str | pathlib.Path | None = None,
                 source_column_name: str = "source_text",
                 include_pattern_names: bool = False):
        """
        Initialize with regex patterns to detect sensitive data

        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        super().__init__(config)
        self.source_column_name = source_column_name

        self._df_class = get_df_class(config.execution_mode)
        self._df_pkg = get_df_pkg(config.execution_mode)
        self._include_pattern_names = include_pattern_names

        if patterns is None:
            if patterns_file is None:
                raise ValueError("Either 'patterns' or 'patterns_file' must be provided")
            patterns = self.load_regex_patterns(patterns_file)
            logger.info("Loaded %d regex pattern groups", len(patterns))

        self.combined_patterns = {}
        # For each entity type, combine multiple patterns into a single regex
        for pattern_name, pattern_list in patterns.items():

            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
            else:
                combined_pattern = pattern_list[0]

            self.combined_patterns[pattern_name] = combined_pattern

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
    def patterns(self) -> dict[str, str]:
        """
        Returns the compiled regex patterns used for detection.
        """
        return self.combined_patterns.copy()

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Scan text for sensitive data using regex patterns
        """

        with msg.payload().mutable_dataframe() as df:
            # Extract the text column to process
            text_series = df[self.source_column_name]

            boolean_columns = {}
            label_columns = []
            for pattern_name, pattern in self.combined_patterns.items():
                bool_col = text_series.str.contains(pattern)
                boolean_columns[pattern_name] = bool_col

                if self._include_pattern_names:
                    label = self._df_pkg.Series(data=None, index=df.index, dtype="O")
                    label[bool_col] = pattern_name
                    label_columns.append(label)

            # Combine all boolean columns into a single series
            bool_df = self._df_class(boolean_columns)
            bool_series = bool_df.any(axis=1)

            if self._include_pattern_names:
                label_series = self._df_pkg.Series(data=None, index=df.index, dtype="O")
                for label_col in label_columns:
                    label_series = label_series.str.cat(label_col, sep=" ", na_rep="").str.strip()

                label_series = label_series.str.replace(" ", ", ")
                df["labels"] = label_series

            # drop input rows that did not match any pattern
            df.drop(
                bool_series[(bool_series == False)].index,  # noqa: E712 pylint: disable=singleton-comparison
                axis=0,
                inplace=True)
            df.reset_index(drop=True, inplace=True)

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
