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
import numpy as np
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
        "street_address": 60,
        "email": 40,
        "phone_number": 45,
        "ipv4": 30,
        "ipv6": 30,
        "date": 20,
        "date_time": 20,
        "time": 20,
        "api_key": 80,
        "customer_id": 65,
        "health_plan_beneficiary_number": 75,
        "medical_record_number": 75
    }

    _NEW_COLUMNS = {
        "risk_score": 0,
        "risk_level": '',
        "data_types_found": np.nan,
        "highest_confidence": 0.0,
        "num_minimal": 0,
        "num_low": 0,
        "num_medium": 0,
        "num_high": 0,
        "num_critical": 0
    }

    def __init__(self,
                 config: Config,
                 *,
                 findings_column: str,
                 type_weights: dict[str, int] | None = None,
                 default_weight: int = 50):
        """Initialize with configuration for risk scoring"""
        super().__init__(config)

        if type_weights is not None:
            self.type_weights = type_weights
        else:
            self.type_weights = self.DEFAULT_TYPE_WEIGHTS.copy()

        # Default weight if type not in dictionary
        self.default_weight = default_weight

        self._findings_column = findings_column
        self._df_pkg = get_df_pkg(config.execution_mode)
        self._group_cols = [self._findings_column] + list(self._NEW_COLUMNS.keys())

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
            return "critical"

        if risk_score >= 60:
            return "high"

        if risk_score >= 40:
            return "medium"

        if risk_score >= 20:
            return "low"

        return "minimal"

    def _score_fn(self, group_df: pd.DataFrame) -> pd.Series | None:

        findings = group_df[self._findings_column]

        if findings is None:
            return None

        flat_findings = []
        for finding in findings:
            if isinstance(finding, str):
                flat_findings.extend(s.strip() for s in finding.split(','))
            else:
                flat_findings.extend(finding)

        findings = flat_findings

        if len(findings) == 0:
            return None

        # Calculate total weighted score
        total_score = 0
        score_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0, "minimal": 0}

        data_types_found = set()
        highest_confidence = 0

        for finding in findings:
            # When `finding` is a dict it came from the GliNER processor, if not then it was bypassed
            if isinstance(finding, dict):
                data_type: str = finding["label"]

                # Adjust by confidence
                confidence = finding["score"]
            else:
                data_type = finding
                confidence = 1.0

            data_types_found.add(data_type)

            # Get weight for this data type
            weight = self.type_weights.get(data_type, self.default_weight)

            if confidence > highest_confidence:
                highest_confidence = confidence

            weighted_score = weight * confidence
            total_score += weighted_score

            # Count by severity
            score_counts[self._risk_score_to_level(weight)] += 1

        # Normalize to 0-100 scale with diminishing returns for many findings
        max_score = 100

        # Calculate normalized risk score
        risk_score = round(min(max_score, total_score / len(findings)))

        # Determine risk level from score
        risk_level = self._risk_score_to_level(risk_score).title()

        df_data = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "data_types_found": sorted(data_types_found),
            "highest_confidence": highest_confidence,
            self._findings_column: findings
        }

        df_data.update({f"num_{level}": count for (level, count) in score_counts.items()})

        return pd.Series(df_data)

    def score(self, msg: ControlMessage) -> ControlMessage:
        """
        Calculate risk scores based on findings
        """

        with msg.payload().mutable_dataframe() as df:
            is_pandas = isinstance(df, pd.DataFrame)
            if not is_pandas:
                df = df.to_pandas()

        df = df.assign(**self._NEW_COLUMNS)
        groups = df.groupby(["original_source_index"], as_index=False)
        result_df = groups[self._group_cols].apply(self._score_fn)

        if not is_pandas:
            result_df = self._df_pkg.from_pandas(result_df)

        msg.payload(MessageMeta(result_df))

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.score))
        builder.make_edge(input_node, node)

        return node
