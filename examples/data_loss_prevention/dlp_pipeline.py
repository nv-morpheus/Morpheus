#!/usr/bin/env python3
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
"""
DLP Pipeline Prototype
A modular Data Loss Prevention pipeline combining regex patterns and SLM classification
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

import re
from gliner import GLiNER


class DLPInputProcessor:
    """Handles input text processing and normalization for DLP pipeline"""
    
    def __init__(self, chunking_size: int = 1000, split_by_paragraphs: bool = False):
        self.chunking_size = chunking_size
        self.split_by_paragraphs = split_by_paragraphs
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess input text:
        1. Normalize whitespace
        2. Split into manageable chunks for processing
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
            
            return chunks
        else:
            return [normalized_text]


class RegexProcessor:
    """Process text with regex patterns to identify structured sensitive data"""
    
    def __init__(self, patterns: Dict[str, List[str]], case_sensitive: bool = False):
        """
        Initialize with regex patterns to detect sensitive data
        
        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        # Optimization: Compile and combine patterns
        self.compiled_patterns = {}
        self.combined_patterns = {}
       # flags = 0 if case_sensitive else re.IGNORECASE
        
        # For each entity type, combine multiple patterns into a single regex
        for entity_type, pattern_list in patterns.items():
       
            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
                self.combined_patterns[entity_type] = re.compile(
                    combined_pattern
                )
            else:
                self.combined_patterns[entity_type] = re.compile(
                    pattern_list[0] #, flags
                )
        
    def process(self, text: str) -> List[Dict[str, Any]]:
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
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize with configuration for SLM-based detection
        
        Args:
            confidence_threshold: Minimum confidence score to report a finding
        """
        self.confidence_threshold = confidence_threshold
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
        self.model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-small-v1.0")
        
    
    def process(self, text: str, regex_findings: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Analyze text using an entity prediction model for sensitive data detection
        
        Args:
            text: The text to analyze
            regex_findings: Optional list of regex findings to filter candidates for classification
            
        Returns:
            List of findings with metadata
        """
        findings = []
        
        # If regex findings are provided, use them to filter text for analysis
        if regex_findings and len(regex_findings) > 0:
            # Extract contexts around regex findings for focused analysis
            contexts = self._extract_contexts_from_regex_findings(text, regex_findings)
            
            # Process each context with the model
            all_entities = []
            for context in contexts:
                # Use the model to predict entities in this context
                context_entities = self.model.predict_entities(
                    context, 
                    self.entity_labels, 
                    threshold=self.confidence_threshold
                )
                all_entities.extend(context_entities)
       
        # Convert model predictions to findings format
        for entity in all_entities:
            findings.append({
                "label": entity["label"],
                "match": entity["text"],
                "span": (entity["start"], entity["end"]),
                "detection_method": "semantic",
                "confidence": entity["score"]
            })
        
        return findings
    
    def _extract_contexts_from_regex_findings(self, text: str, regex_findings: List[Dict[str, Any]]) -> List[str]:
        """
        Extract text contexts around regex matches to focus SLM analysis
        
        Args:
            text: The full text being analyzed
            regex_findings: List of regex findings with span information
            
        Returns:
            List of text contexts for focused analysis
        """
        contexts = []
        context_window = 100  # Characters before and after the match
        
        for finding in regex_findings:
            if finding.get("span"):
                start, end = finding["span"]
                
                # Expand the context window
                context_start = max(0, start - context_window)
                context_end = min(len(text), end + context_window)
                
                # Extract the context
                context = text[context_start:context_end]
                contexts.append(context)
        
        # If no valid contexts were extracted, use the full text
        if not contexts:
            contexts.append(text)
            
        return contexts

class ResultAggregator:

    """Combines and refines results from different processors"""
    
    def __init__(self, semantic_to_data_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize with configuration for result aggregation
        
        Args:
            semantic_to_data_mapping: Maps semantic categories to specific data types
        """
        # Default mapping if none provided
        self.semantic_to_data_mapping = semantic_to_data_mapping or {
            "password": "password",
            "credit_card": "credit_card",
            "ssn": "ssn",
            "address": "address",
            "email": "email",
            "phone_us": "phone_numbers",
            "phone_numbers": "phone_numbers",
            "ip_address": "ip_address",
            "date": "date",
            "api_key": "api_key",
            "api_credentials": "api_credentials",
            "customer_id": "customer_id",
            # Semantic categories
            "personal": "personal",
            "financial": "financial",
            "health": "health"
        }
    
    def aggregate(self, semantic_findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate and deduplicate findings from different processors
        
        Returns:
            Combined list of findings
        """
        # Only use semantic findings
        all_findings = []
        
        # Keep track of spans that have already been covered by semantic findings
        covered_spans = []
        
        # Process semantic findings with deduplication
        for finding in semantic_findings:
            # Skip if this finding has no span information
            if not finding.get("span"):
                continue
            
            # Check if this span overlaps with any existing finding
            semantic_span = finding["span"]
            overlap = False
            
            for existing_span in covered_spans:
                # Check for any overlap between spans
                if (semantic_span[0] <= existing_span[1] and 
                    semantic_span[1] >= existing_span[0]):
                    overlap = True
                    break
            
            # If no overlap, add this finding
            if not overlap:
                # Map the semantic type to a data type if needed
                finding_type = finding["label"]
                if finding_type in self.semantic_to_data_mapping:
                    finding["data_type"] = self.semantic_to_data_mapping[finding_type]
                else:
                    finding["data_type"] = finding_type
                    
                all_findings.append(finding)
                covered_spans.append(semantic_span)
        
        return all_findings


class RiskScorer:
    """Analyzes findings to calculate risk scores and metrics"""
    
    def __init__(self):
        """Initialize with configuration for risk scoring"""
        # Weights for different types of sensitive data (0-100 scale)
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
            "customer_id": 65,
            # Semantic categories
            "personal": 70,
            "financial": 85,
            "health": 75,
            "api_credentials": 75
        }
        
        # Default weight if type not in dictionary
        self.default_weight = 50
    
    def score(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate risk scores based on findings
        
        Returns:
            Risk scoring results and metrics
        """
        if not findings:
            return {
                "risk_score": 0,
                "risk_level": "None",
                "data_types_found": [],
                "highest_confidence": 0.0,
                "severity_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        
        # Calculate total weighted score
        total_score = 0
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        print(findings[0])
        
        for finding in findings:
            # Get data type (either direct type or mapped from semantic)
            data_type = finding.get("data_type", finding["label"])
            
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
        data_types_found = list({finding.get("data_type", finding["label"]) 
                               for finding in findings})
        
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
    """Main DLP pipeline that orchestrates the processing stages"""
    
    def __init__(self, 
                regex_patterns: dict[str, list[str]], 
                confidence_threshold: float = 0.7,
                case_sensitive: bool = False):
        """
        Initialize the DLP pipeline with its component processors
        
        Args:
            regex_patterns: Dictionary mapping data types to regex patterns
            confidence_threshold: Minimum confidence for SLM findings
            case_sensitive: Whether regex matching should be case sensitive
        """
        # if regex_file:
        #     with open(regex_file, 'r') as f:
        #         regex_patterns_file = json.load(f)
        
        self.input_processor = DLPInputProcessor()
        self.regex_processor = RegexProcessor(patterns=regex_patterns, case_sensitive=case_sensitive)
        self.slm_processor = GliNERProcessor(confidence_threshold) 
        self.result_aggregator = ResultAggregator()
        self.risk_scorer = RiskScorer()
    
    def process(self, document: str) -> Dict[str, Any]:
        """
        Process a document through the entire DLP pipeline
        
        Args:
            document: The text content to analyze
            
        Returns:
            Complete DLP analysis results
        """
        start_time = time.time()
        
        # Stage 1: Input processing
        text_chunks = self.input_processor.preprocess(document)
        
        all_findings = []
        
        # Process each chunk
        for chunk in text_chunks:
            # Stage 2: Regex processing
            regex_findings = self.regex_processor.process(chunk)
            
            # Stage 3: SLM processing - pass regex findings to filter candidates
            semantic_findings = self.slm_processor.process(chunk, regex_findings)
            
            # Stage 4: Result aggregation
            chunk_findings = self.result_aggregator.aggregate(
               semantic_findings
            )
            
            all_findings.extend(chunk_findings)
        
        # Stage 5: Risk scoring
        risk_assessment = self.risk_scorer.score(all_findings)
        
        # Prepare final result
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "findings": all_findings,
            "total_findings": len(all_findings),
            "risk_assessment": risk_assessment,
            "processing_time": processing_time,
            "document_length": len(document),
            "processing_rate": len(document) / processing_time if processing_time > 0 else 0
        }


def format_findings(results: Dict[str, Any]) -> str:
    """Format DLP findings as a readable report"""
    output = []
    output.append("===== DLP Analysis Report =====")
    output.append(f"Risk Level: {results['risk_assessment']['risk_level']} ({results['risk_assessment']['risk_score']}/100)")
    output.append(f"Total Findings: {results['total_findings']}")
   # output.append(f"Processing Time: {results['processing_time']:.3f} seconds")
    
    output.append("\nSensitive Data Types Found:")
    for data_type in results['risk_assessment']['data_types_found']:
        output.append(f"  - {data_type}")
    
    output.append("\nSeverity Distribution:")
    dist = results['risk_assessment']['severity_distribution']
    output.append(f"  High: {dist['high']}, Medium: {dist['medium']}, Low: {dist['low']}")
    
    if results['findings']:
        output.append("\nDetailed Findings:")
        for i, finding in enumerate(results['findings'][:10], 1):  # Limit to first 10 findings
            output.append(f"\n  Finding {i}:")
            output.append(f"    Type: {finding['label']}")
           # output.append(f"    Method: {finding['detection_method']}")
            output.append(f"    Confidence: {finding['score']:.2f}")
            if finding['text']:
                # Truncate and sanitize match text if needed
                match_text = finding['text']
                if len(match_text) > 40:
                    match_text = match_text[:37] + "..."
                output.append(f"    Match: {match_text}")
    
        if len(results['findings']) > 10:
            output.append(f"\n  ... and {len(results['findings']) - 10} more findings")
    
    return "\n".join(output)


def test_dlp_pipeline():
    """Test the DLP pipeline with example documents"""
    
    
    regex_file = "regex_patterns.json"
    with open(regex_file, 'r') as f:
        regex_patterns = json.load(f)
        
    # original_pipeline = OriginalDLPPipeline(REGEX_PATTERNS)
    dlp_pipeline = DLPPipeline(regex_patterns)
    
    test_documents = [
        {
            "title": "Config File with Keys",
            "content": """
# Production API Configuration
api_key = "ak_live_HJd8e7h23hFxMznWcQE5TWqL"
api_secret = "sk_test_abcdefghijklmnopqrstuvwxyz12345"
debug = false

# Database Connection
DB_HOST = "db.example.com"
DB_USER = "admin"
DB_PASSWORD = "SecurePassword123!"
            """
        },
        {
            "title": "Patient Information",
            "content": """
CONFIDENTIAL PATIENT INFORMATION

Patient: John Smith
DOB: 05/12/1978
SSN: 123-45-6789
Contact: (555) 123-4567

Medical History:
- Hypertension diagnosed 2015
- Allergic to penicillin
- Regular checkups every 6 months
            """
        },
        {
            "title": "Order Receipt",
            "content": """
PURCHASE RECEIPT

Customer: Jane Doe
Email: jane.doe@example.com
Card: VISA ending in 4567
Transaction: $128.50

Shipping Address:
123 Main Street
Anytown, CA 94538
            """
        },
        {
            "title": "Non-Sensitive Document",
            "content": """
PROJECT TIMELINE

Phase 1: Planning (Week 1-2)
Phase 2: Development (Week 3-8)
Phase 3: Testing (Week 9-10)
Phase 4: Deployment (Week 11-12)

Team meeting every Monday at 10am.
            """
        }
    ]
    
    print("===== DLP Pipeline Test =====\n")
    
    for doc in test_documents:
        print(f"Document: {doc['title']}")
        print("-" * 40)
        
        results = dlp_pipeline.process(doc['content'])
        report = format_findings(results)
        print(report)
        
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Test both pipelines and compare results
    test_dlp_pipeline()