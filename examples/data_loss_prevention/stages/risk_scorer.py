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

import math
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_utils import get_df_pkg


@register_stage("risk-scorer")
class RiskScorer(GpuAndCpuMixin, ControlMessageStage):
    """Analyzes findings to calculate risk scores and metrics"""

    DEFAULT_TYPE_WEIGHTS = {
        "password": 85,
        "credit_card_number": 90,
        "ssn": 95,
        "address": 60,
        "email": 40,
        "phone_number": 45,
        "ipv4": 30,
        "ipv6": 30,
        "date": 20,
        "date_time": 20,
        "time": 20,
        "api_key": 80,
        "customer_id": 65,
        "personal": 70,
        "financial": 85,
        "health_plan_beneficiary_number": 75,
        "medical_record_number": 75,
        "api_credentials": 75
    }

    def __init__(self, config: Config, *, type_weights: dict[str, int] | None = None, default_weight: int = 50):
        """Initialize with configuration for risk scoring"""
        super().__init__(config)

        if type_weights is not None:
            self.type_weights = type_weights
        else:
            self.type_weights = self.DEFAULT_TYPE_WEIGHTS.copy()

        # Default weight if type not in dictionary
        self.default_weight = default_weight

        self._df_pkg = get_df_pkg(config.execution_mode)

    @property
    def name(self) -> str:
        return "risk-scorer"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @staticmethod
    def _risk_score_to_level(risk_score: int) -> str:
        """Convert risk score to risk level string"""
        if risk_score >= 80:
            return "Critical"

        if risk_score >= 60:
            return "High"

        if risk_score >= 40:
            return "Medium"

        if risk_score >= 20:
            return "Low"

        return "Minimal"

    def _score_fn(self, row_index: int, group_df: pd.DataFrame) -> pd.DataFrame | None:

        findings = group_df.dlp_findings

        if findings is None:
            return None

        flat_findings = []
        for finding in findings:
            flat_findings.extend(finding)

        findings = flat_findings
        if len(findings) == 0:
            return None

        # Calculate total weighted score
        total_score = 0
        num_high = 0
        num_medium = 0
        num_low = 0

        data_types_found = set()
        highest_confidence = 0

        for finding in findings:
            # Get data type (either direct type or mapped from semantic)
            data_type: str = finding["label"]
            data_types_found.add(data_type)

            # Get weight for this data type
            weight = self.type_weights.get(data_type, self.default_weight)

            # Adjust by confidence
            confidence = finding["score"]

            if confidence > highest_confidence:
                highest_confidence = confidence

            weighted_score = weight * confidence
            total_score += weighted_score

            # Count by severity
            if weighted_score >= 80:
                num_high += 1
            elif weighted_score >= 50:
                num_medium += 1
            else:
                num_low += 1

        # Normalize to 0-100 scale with diminishing returns for many findings
        max_score = 100
        normalization_factor = max(1, math.log2(len(findings) + 1)) * 20  # Adjust scaling factor

        # Calculate normalized risk score
        risk_score = round(min(max_score, total_score / normalization_factor))

        # Determine risk level from score
        risk_level = self._risk_score_to_level(risk_score)

        return pd.DataFrame({
            "original_source_index": row_index,
            "risk_score": [risk_score],
            "risk_level": [risk_level],
            "data_types_found": [sorted(data_types_found)],
            "highest_confidence": [highest_confidence],
            "num_high": [num_high],
            "num_medium": [num_medium],
            "num_low": [num_low],
            "dlp_findings": [findings]
        })

    def score(self, msg: ControlMessage) -> ControlMessage:
        """
        Calculate risk scores based on findings
        """

        with msg.payload().mutable_dataframe() as df:
            is_pandas = isinstance(df, pd.DataFrame)
            if not is_pandas:
                df = df.to_pandas()

        groups = df.groupby(["original_source_index"], as_index=False)
        results = []
        for (original_source_index, group_df) in groups:
            scored_df = self._score_fn(original_source_index, group_df)
            if scored_df is not None:
                results.append(scored_df)

        result_df = pd.concat(results, axis=0)

        if not is_pandas:
            result_df = self._df_pkg.from_pandas(result_df)

        msg.payload(MessageMeta(result_df))

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.score))
        builder.make_edge(input_node, node)

        return node
