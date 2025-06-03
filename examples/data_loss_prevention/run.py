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

# # DLP Pipeline Inference Workflow
#
# This notebook demonstrates a Data Loss Prevention (DLP) pipeline that combines regex pattern matching with GliNER model inference for entity detection.

import json
import os
import time

from regex_processor import GliNERProcessor
from regex_processor import RegexProcessor

# ## 1. Dataset creation and processing

from dataset_creation import load_and_process_datasets

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

gretel_dataset = load_and_process_datasets(['gretel'], num_samples=2000)

gretel_dataset.head()

# ## 2. Load and Configure Regex Patterns
#
# First, we'll load the regex patterns from the benchmark patterns file.

# Load regex patterns
regex_file = os.path.join(CUR_DIR, "data/regex_patterns.json")

if not os.path.exists(regex_file):
    # Create a basic set of patterns as fallback
    patterns = {
        "credit_card_number": [
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11}|(?:\d{4}[-\s]?){3}\d{4}|\d{16})\b"
        ],
        "ssn": [r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"],
        "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"],
        "phone_number": [r"\b(?:\+?1[-\.\s]?)?(?:\(\d{3}\)|\d{3})[-\.\s]?\d{3}[-\.\s]?\d{4}\b|\b\d{10}\b"],
        "ipv4": [r"\b(?:\d{1,3}\.){3}\d{1,3}\b"],
        "date": [r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"]
    }
else:
    with open(regex_file, 'r') as f:
        patterns = json.load(f)

print(f"Loaded {len(patterns)} regex pattern groups")

# ## 3. DLP Pipeline
#
# Now we'll create a  DLP pipeline that integrates the regex pipeline and Gliner SLM


class DLPInputProcessor:
    """Handles input text processing and normalization for DLP pipeline"""

    def __init__(self, chunking_size: int = 1000, split_by_paragraphs: bool = False):
        self.chunking_size = chunking_size
        self.split_by_paragraphs = split_by_paragraphs

    def preprocess(self, text: str) -> list[str]:
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

    def score(self, findings: list[dict[str, list]]) -> dict[str, list]:
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
                "severity_distribution": {
                    "high": 0, "medium": 0, "low": 0
                }
            }

        # Calculate total weighted score
        total_score = 0
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
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


class DLPPipeline:
    """ DLP pipeline integrating components"""

    def __init__(self,
                 regex_patterns: dict[str, list[str]],
                 confidence_threshold: float = 0.7,
                 model_name: str = "gretelai/gretel-gliner-bi-small-v1.0",
                 context_window: int = 300):
        """Initialize the enhanced DLP pipeline"""
        self.input_processor = DLPInputProcessor(split_by_paragraphs=False)
        self.regex_processor = RegexProcessor(patterns=regex_patterns)
        self.gliner_processor = GliNERProcessor(confidence_threshold=confidence_threshold,
                                                context_window=context_window,
                                                model_name=model_name)
        self.risk_scorer = RiskScorer()

    def inference(self, document: str, failback: bool = False) -> dict[str, list]:
        """Process a document through the DLP pipeline"""
        regex_findings = self.regex_processor.process(document)
        return self.gliner_processor.process(document, regex_findings, failback=failback)

    def process(self, document: str) -> dict[str, list]:
        """Process a document through the DLP pipeline"""
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


# ## 4. Pipeline processing
#
#
#

test_documents = [{
    "title":
        "Patient Information",
    "content":
        """
PATIENT INFORMATION
Medical Record #: MRN-12345678
Name: John Smith
DOB: 01/15/1985
SSN: 123-45-6789
Address: 5678 Pine Avenue, Apt 302, Chicago, IL 60601
Phone: 773-555-1234
Email: jsmith@email.net
Insurance ID: INS-987654321

VISIT SUMMARY
Date: 03/22/2023
Time: 14:30
Provider: Dr. Sarah Johnson, MD (NPI: 1234567890)
Diagnosis: Hypertension (ICD-10: I10)
Medications: Lisinopril 10mg daily
Follow-up: 3 months
"""
},
                  {
                      "title":
                          "Config File with API Keys",
                      "content":
                          """
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
                      "title":
                          "Order Receipt",
                      "content":
                          """
PURCHASE RECEIPT

Customer: Jane Doe
Email: jane.doe@example.com
Card: VISA ending in 4567
Transaction: $128.50

Shipping Address:
123 Main Street
Anytown, CA 94538
"""
                  }]

# ### Create and Initialize the DLP Pipeline
#
# Now we'll create the pipeline with the components for inference on sample dataset

# Create the enhanced DLP pipeline
dlp_pipeline = DLPPipeline(regex_patterns=patterns,
                           confidence_threshold=0.7,
                           model_name="gretelai/gretel-gliner-bi-small-v1.0",
                           context_window=300)


# ##### Running batch of examples


def batch_process_documents(pipeline, documents):
    """Process multiple documents and aggregate results"""
    all_results = []
    total_findings = 0
    total_processing_time = 0
    total_document_length = 0

    for doc in documents:

        result = pipeline.process(doc)
        all_results.append(result.get('all_findings', []))

        total_findings += result['total_findings']
        total_processing_time += result['performance_metrics']['total_time']
        total_document_length += result['document_length']

    # Aggregate metrics
    avg_processing_time = total_processing_time / len(documents) if documents else 0
    avg_findings_per_doc = total_findings / len(documents) if documents else 0
    overall_throughput = total_document_length / total_processing_time if total_processing_time > 0 else 0

    # Print summary
    print(f"\n{'='*50}")
    print(f"Batch Processing Summary")
    print(f"{'='*50}")
    print(f"Documents processed: {len(documents)}")
    print(f"Total findings: {total_findings}")
    print(f"Average findings per document: {avg_findings_per_doc:.2f}")
    print(f"Total processing time: {total_processing_time:.3f} seconds")
    print(f"Average processing time: {avg_processing_time:.3f} seconds per document")
    print(f"Overall throughput: {overall_throughput:.2f} characters/second")

    return all_results


batch_results = batch_process_documents(dlp_pipeline, gretel_dataset['source_text'].tolist())

