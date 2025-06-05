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


class RiskScorer:
    """Analyzes findings to calculate risk scores and metrics"""

    def __init__(self):
        """Initialize with configuration for risk scoring"""

        self.type_weights = {
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

        # Default weight if type not in dictionary
        self.default_weight = 50

    def score(self, findings: list[dict[str, list]]) -> dict[str, typing.Any]:
        """
        Calculate risk scores based on findings

        Returns:
            Risk scoring results and metrics
        """
        if not findings:
            return {
                "risk_score": 0,
                "risk_level": None,
                "data_types_found": [],
                "highest_confidence": 0.0,
                "severity_distribution": {
                    "high": 0, "medium": 0, "low": 0
                }
            }

        # Calculate total weighted score
        total_score = 0
        severity_counts = {"high": 0, "medium": 0, "low": 0}

        for finding in findings:
            # Get data type (either direct type or mapped from semantic)
            data_type: str = finding.get("data_type", finding["label"])

            # Get weight for this data type
            weight = self.type_weights.get(data_type, self.default_weight)

            # Adjust by confidence
            confidence = finding["score"]
            weighted_score = weight * confidence
            total_score += weighted_score

            # Count by severity
            if weight >= 80:
                severity_counts["high"] += 1
            elif weight >= 50:
                severity_counts["medium"] += 1
            else:
                severity_counts["low"] += 1

        # Normalize to 0-100 scale with diminishing returns for many findings
        import math
        max_score = 100
        normalization_factor = max(1, math.log2(len(findings) + 1)) * 20  # Adjust scaling factor

        # Calculate normalized risk score
        risk_score = min(max_score, total_score / normalization_factor)

        # Determine risk level from score
        risk_level = "Critical" if risk_score >= 80 else \
                     "High" if risk_score >= 60 else \
                     "Medium" if risk_score >= 40 else \
                     "Low" if risk_score >= 20 else "Minimal"

        # Get unique data types found
        data_types_found = list({finding.get("data_type", finding["label"]) for finding in findings})

        # Find highest confidence score
        highest_confidence = max(finding["score"] for finding in findings)

        return {
            "risk_score": int(risk_score),
            "risk_level": risk_level,
            "data_types_found": data_types_found,
            "highest_confidence": highest_confidence,
            "severity_distribution": severity_counts
        }
