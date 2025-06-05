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
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin


@register_stage("risk-scorer")
class RiskScorer(ControlMessageStage, GpuAndCpuMixin):
    """Analyzes findings to calculate risk scores and metrics"""

    DEFAULT_TYPE_WEIGHTS = {
        "password": 85,
        "credit_card": 90,
        "ssn": 95,
        "address": 60,
        "email": 40,
        "phone_us": 45,
        "phone_numbers": 45,
        "ip_address": 30,
        "date": 20,
        "api_key": 80,
        "customer_id": 65,  # Semantic categories
        "personal": 70,
        "financial": 85,
        "health": 75,
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
        elif risk_score >= 60:
            return "High"
        elif risk_score >= 40:
            return "Medium"
        elif risk_score >= 20:
            return "Low"

        return "Minimal"

    def _score_row(self, findings: list[dict[str, typing.Any]] | None) -> dict[str, typing.Any]:

        if findings is None or len(findings) == 0:
            return {
                "risk_score": 0,
                "risk_level": None,
                "data_types_found": [],
                "highest_confidence": 0.0,
                "num_high": 0,
                "num_medium": 0,
                "num_low": 0
            }

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

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "data_types_found": sorted(data_types_found),
            "highest_confidence": highest_confidence,
            "num_high": num_high,
            "num_medium": num_medium,
            "num_low": num_low
        }

    def score(self, msg: ControlMessage) -> ControlMessage:
        """
        Calculate risk scores based on findings
        """

        with msg.payload().mutable_dataframe() as df:
            dlp_findings = df['dlp_findings']
            if not isinstance(dlp_findings, pd.Series):

                # cudf series doesn't support iteration
                dlp_findings = dlp_findings.to_arrow().to_pylist()

            scores = {
                "risk_score": [],
                "risk_level": [],
                "data_types_found": [],
                "highest_confidence": [],
                "num_high": [],
                "num_medium": [],
                "num_low": [],
            }
            for findings in dlp_findings:
                score = self._score_row(findings)
                for (key, value) in score.items():
                    scores[key].append(value)

            for key, value in scores.items():
                df[key] = value

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.score))
        builder.make_edge(input_node, node)

        return node
