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
import math
import time

from regex_processor import GliNERProcessor
from regex_processor import RegexProcessor


class DLPInputProcessor:
    """Handles input text processing and normalization for DLP pipeline"""

    def __init__(self, chunking_size: int = 1000, split_by_paragraphs: bool = False):
        self.chunking_size = chunking_size
        self.split_by_paragraphs = split_by_paragraphs

    def preprocess(self, text: str) -> list[str]:
        """Preprocess input text.

        Parameters
        ----------
        text : str
            The text to preprocess.

        Returns
        -------
        list[str]
            A list of preprocessed text chunks.
        """
        # Basic normalization
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

        if self.split_by_paragraphs:
            # For larger texts, split into chunks to optimize processing
            if len(normalized_text) > self.chunking_size:
                chunks = []
            # Split by paragraphs first to preserve content boundaries
            paragraphs = normalized_text.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > self.chunking_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para

            if current_chunk:
                chunks.append(current_chunk)
            normalized_text = chunks
        else:
            normalized_text = [normalized_text]
        return normalized_text


class RiskScorer:

    def __init__(self, type_weights: dict[str, int] = None):
        """Initialize with configuration for risk scoring.

        Parameters
        ----------
        type_weights : dict[str, int], optional
            A dictionary of type weights for the risk scoring.
        """
        self.type_weights = type_weights
        if type_weights is None:
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

    def score(self, findings: list[dict[str, list]]) -> dict[str, list]:
        """
        Calculate risk scores based on findings.

        Parameters
        ----------
        findings : list[dict[str, list]]
            A list of findings from the DLP pipeline.

        Returns:
            Risk scoring results and metrics
        """
        if not findings:
            return {
                "risk_score": 0,
                "risk_level": "None",
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
            data_type = finding.get("label")

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
        # max_score = 100
        max_score = 100
        # Adjust scaling factor to increase risk for more findings
        normalization_factor = max(1, math.log2(len(findings) + 1)) * 2
        # Calculate normalized risk score
        risk_score = min(max_score, total_score / normalization_factor)

        # Determine risk level from score
        risk_level = "Critical" if risk_score >= 80 else \
                     "High" if risk_score >= 60 else \
                     "Medium" if risk_score >= 40 else \
                     "Low" if risk_score >= 20 else "Minimal"

        # Get unique data types found
        data_types_found = list({finding.get("label") for finding in findings})

        # Find highest confidence score
        highest_confidence = max(finding["score"] for finding in findings)

        return {
            "risk_score": int(risk_score),
            "risk_level": risk_level,
            "data_types_found": data_types_found,
            "highest_confidence": highest_confidence,
            "severity_distribution": severity_counts
        }


class DLPPipeline:
    """ DLP pipeline integrating components.
    This class is used to process a document through the DLP pipeline.
    It integrates the RegexProcessor, GliNERProcessor, and RiskScorer components.
    """

    def __init__(self,
                 *,
                 regex_patterns: dict[str, list[str]],
                 confidence_threshold: float = 0.3,
                 model_name: str = "gretelai/gretel-gliner-bi-small-v1.0",
                 context_window: int = 300,
                 config_file: str = "data/config.json"):
        """Initialize the enhanced DLP pipeline

        Parameters
        ----------
        regex_patterns : dict[str, list[str]]
            A dictionary of regex patterns for the DLP pipeline.
        confidence_threshold : float, optional
            The confidence threshold for the DLP pipeline, by default 0.3
        model_name : str, optional
            The name of the GLiNER model to use, by default "gretelai/gretel-gliner-bi-small-v1.0"
        context_window : int, optional
            The context window for the GLiNER model, by default 300
        config_file : str, optional
            The path to the config file for the GLiNER model, by default "data/config.json"
        """

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.input_processor = DLPInputProcessor(split_by_paragraphs=False)
        self.regex_processor = RegexProcessor(patterns=regex_patterns)
        # self.regex_processor = GPURegexEntityDetector(patterns=regex_patterns)
        self.gliner_processor = GliNERProcessor(confidence_threshold=confidence_threshold,
                                                context_window=context_window,
                                                model_name=model_name,
                                                labels=config['entity_labels'])
        self.risk_scorer = RiskScorer(type_weights=config['type_weights'])

    def inference(self, document: str, failback: bool = False) -> dict[str, list]:
        """Process a document through the DLP pipeline.

        Parameters
        ----------
        document : str
            The document to process.
        failback : bool, optional
            If True, the GliNER processor will failback to the Regex processor if no entities are found.
            by default False

        Returns
        -------
        dict[str, list]
            A dictionary containing the findings from the DLP pipeline.
        """
        regex_findings = self.regex_processor.process(document)
        return self.gliner_processor.process(document, regex_findings, failback=failback)

    def process(self, document: str) -> dict[str, list]:
        """Process a document through the DLP pipeline.

        Parameters
        ----------
        document : str
            The document to process.

        Returns
        -------
        dict[str, list]
            A dictionary containing the findings from the DLP pipeline.
        """
        start_time = time.time()

        # Stage 1: Input processing
        text_chunks = self.input_processor.preprocess(document)

        # Process metrics
        regex_times = []
        gliner_times = []
        all_findings = []

        # Process each chunk
        for chunk in text_chunks:
            # Stage 2: Regex processing with timing
            regex_start = time.time()
            regex_findings = self.regex_processor.process(chunk)
            regex_time = time.time() - regex_start
            regex_times.append(regex_time)

            # Stage 3: GLiNER processing with timing
            gliner_start = time.time()
            semantic_findings = self.gliner_processor.process(chunk, regex_findings)
            gliner_time = time.time() - gliner_start
            gliner_times.append(gliner_time)

            all_findings.extend(semantic_findings)

        # Stage 5: Risk scoring
        risk_assessment = self.risk_scorer.score(all_findings)

        # Calculate performance metrics
        end_time = time.time()
        total_processing_time = end_time - start_time
        total_regex_time = sum(regex_times)
        total_gliner_time = sum(gliner_times)

        # Create performance report
        performance_metrics = {
            "total_time": total_processing_time,
            "regex_time": total_regex_time,
            "gliner_time": total_gliner_time,
            "regex_percentage": (total_regex_time / total_processing_time) * 100 if total_processing_time > 0 else 0,
            "gliner_percentage": (total_gliner_time / total_processing_time) * 100 if total_processing_time > 0 else 0,
            "throughput": len(document) / total_processing_time if total_processing_time > 0 else 0
        }

        return {
            "findings": all_findings,
            "total_findings": len(all_findings),
            "risk_assessment": risk_assessment,
            "performance_metrics": performance_metrics,
            "document_length": len(document)
        }
