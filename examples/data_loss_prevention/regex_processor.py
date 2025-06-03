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

import cudf
import re2 as re
from gliner import GLiNER


class RegexProcessor:
    """Process text with regex patterns to identify structured sensitive data"""

    def __init__(self, patterns: dict[str, list[str]], case_sensitive: bool = False):
        """
        Initialize with regex patterns to detect sensitive data
        
        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        # Optimization: Compile and combine patterns
        self.compiled_patterns = {}
        self.combined_patterns = {}
        #flags = 0 if case_sensitive else re.IGNORECASE

        # For each entity type, combine multiple patterns into a single regex
        for entity_type, pattern_list in patterns.items():

            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
                self.combined_patterns[entity_type] = re.compile(
                    combined_pattern,  # flags
                )
            else:
                self.combined_patterns[entity_type] = re.compile(
                    pattern_list[0],  # flags
                )

    def process(self, text: str) -> list[dict[str, list]]:
        """
        Scan text for sensitive data using regex patterns
        
        Returns:
            List of findings with metadata
        """
        findings = []

        for pattern_name, pattern in self.combined_patterns.items():
            # for pattern in pattern_list:
            matches = pattern.finditer(text)
            for match in matches:
                findings.append({
                    "label": pattern_name,
                    "match": match.group(),
                    "span": match.span(),
                    "detection_method": "regex",
                    "confidence": 0.9  # High confidence for regex matches
                })

        return findings


class GliNERProcessor:
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text
    """

    def __init__(self,
                 model_name: str = "gretelai/gretel-gliner-bi-small-v1.0",
                 confidence_threshold: float = 0.7,
                 context_window: int = 100,
                 labels: list[str] = None):
        """
        Initialize with configuration for SLM-based detection
        
        Args:
            confidence_threshold: Minimum confidence score to report a finding
        """
        self.confidence_threshold = confidence_threshold
        if labels:
            self.entity_labels = labels
        else:
            self.entity_labels = [
                "medical_record_number",
                "date_of_birth",
                "ssn",
                "date",
                "first_name",
                "email",
                "last_name",
                "customer_id",
                "employee_id",
                "name",
                "street_address",
                "phone_number",
                "ipv4",
                "credit_card_number",
                "license_plate",
                "address",
                "user_name",
                "device_identifier",
                "bank_routing_number",
                "date_time",
                "company_name",
                "unique_identifier",
                "biometric_identifier",
                "account_number",
                "city",
                "certificate_license_number",
                "time",
                "postcode",
                "vehicle_identifier",
                "coordinate",
                "country",
                "api_key",
                "ipv6",
                "password",
                "health_plan_beneficiary_number",
                "national_id",
                "tax_id",
                "url",
                "state",
                "swift_bic",
                "cvv",
                "pin"
            ]

        # Load the fine-tuned GLiNER model
        self.model = GLiNER.from_pretrained(model_name, map_location="cuda")
        self.context_window = context_window

    def process(self,
                text: str,
                regex_findings: list[dict[str, list]] = None,
                failback: bool = True) -> list[dict[str, list]]:
        """
        Analyze text using an entity prediction model for sensitive data detection
        
        Args:
            text: The text to analyze
            regex_findings: Optional list of regex findings to filter candidates for classification
            
        Returns:
            List of findings with metadata
        """

        unique_entities = True

        # If regex findings are provided, use them to filter text for analysis
        if regex_findings and len(regex_findings) > 0:

            contexts, spans = self._extract_contexts_from_regex_findings(text, regex_findings)
            assert len(contexts) == len(spans)
            all_entities = self.model.batch_predict_entities(contexts,
                                                             self.entity_labels,
                                                             flat_ner=True,
                                                             threshold=self.confidence_threshold,
                                                             multi_label=False)
            if unique_entities:
                seen = set()
                unique_entities = []
                for i, entities in enumerate(all_entities):
                    span_offset = spans[i][0]
                    for entity in entities:
                        entity["start"] += span_offset
                        entity["end"] += span_offset
                        entity_key = (entity["label"], entity["text"], entity["start"], entity["end"])
                        if entity_key not in seen:
                            seen.add(entity_key)
                            unique_entities.append(entity)
                all_entities = unique_entities
        elif failback:
            all_entities = self.gliner_predict(text)
            all_entities = self.filter_entities(all_entities)
        else:
            all_entities = []

        return all_entities

    def _extract_contexts_from_regex_findings(self, text: str, regex_findings: list[dict[str, list]]) -> list[str]:
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
        return contexts, spans

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


class GPURegexProcessor:
    """Process text with regex patterns to identify structured sensitive data using GPU acceleration"""

    def __init__(self, patterns: dict[str, list[str]], case_sensitive: bool = False):
        """
        Initialize with regex patterns to detect sensitive data
        
        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        import cudf

        # Optimization: Compile and combine patterns
        self.compiled_patterns = {}
        self.combined_patterns = {}
        self.case_sensitive = case_sensitive

        # For each entity type, combine multiple patterns into a single regex
        for entity_type, pattern_list in patterns.items():
            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
                self.combined_patterns[entity_type] = combined_pattern
            else:
                self.combined_patterns[entity_type] = pattern_list[0]

    def process(self, text: str, delimiter: str = '\n') -> list[dict[str, list]]:
        """
        Scan text for sensitive data using regex patterns with GPU acceleration
        
        Returns:
            List of findings with metadata
        """

        findings = []

        # Convert text to cuDF Series
        text_series = cudf.Series(text.split(delimiter))

        for pattern_name, pattern in self.combined_patterns.items():
            if pattern_name not in self.entity_labels:
                continue

            matches = text_series.str.findall(pattern)

            # Process matches
            if len(matches[0]) > 0:
                for match in matches[0]:
                    # Find the span of the match in the original text
                    start = text.find(match)
                    end = start + len(match)

                    findings.append({
                        "label": pattern_name,
                        "match": match,
                        "span": (start, end),
                        "detection_method": "gpu_regex",
                        "confidence": 0.9  # High confidence for regex matches
                    })

        return findings
