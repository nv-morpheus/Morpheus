# Data Loss Prevention (DLP) Pipeline with Morpheus

This example demonstrates how to use Morpheus to implement a DLP solution that combines regex pattern matching with small language model (SLM) powered semantic analysis for data protection.

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |


## Background

The goal of this  DLP pipeline is to identify and mitigate exposure of sensitive information in enterprise data streams as quickly and accurately as possible. Effective DLP systems are essential in today's enterprise and cloud environments to:

- **Prevent Sensitive Data Leakage**: Identify and mitigate exposure of personally identifiable information (PII), protected health information (PHI), credentials, and intellectual property
- **Ensure Regulatory Compliance**: Meet data protection mandates such as GDPR, HIPAA, PCI-DSS, and ISO 27001
- **Detect Insider Threats**: Monitor and flag suspicious access to sensitive data by internal actors
- **Secure Cloud and SaaS Workloads**: Protect data in motion and at rest across hybrid and multi-cloud architectures

This pipeline implements a hybrid approach that couples:
- **Regex-based pre-filtering** for fast and scalable detection with high recall (1000x speed improvement over AI-only approaches)
- **Contextual entity validation** using GLiNER-based semantic analysis, reducing false positives by 80%

## 🎯 Supported Data Types

The DLP pipeline is capable of detecting multiple categories of sensitive information inlcuding these and more entities:

### Personal Information
1. **SSN**: Social Security Numbers with flexible formatting
2. **Credit Cards**: Visa, MasterCard, Amex, Discover with validation
3. **Phone Numbers**: US/International formats with area codes
4. **Email Addresses**: Comprehensive email pattern matching

### Technical Information  
5. **IP Addresses**: IPv4 & IPV6 addresses and subnets
6. **API Keys**: Various API key formats and tokens
7. **Passwords**: Password pattern detection
8. **URLs**: Web addresses and endpoints

### Healthcare & Financial
9. **Medical Record Numbers**: Healthcare identifiers
10. **Insurance IDs**: Health insurance identifiers
11. **Customer IDs**: Business customer identifiers
12. **Account Numbers**: Bank and financial accounts

### The Dataset

The dataset that this workflow processes can be various text formats including JSON, CSV, or plain text documents converted into textual data. For example, below is a sample document containing multiple types of sensitive information:

```json
{
  "timestamp": 1616380971990,
  "document_type": "patient_record",
  "content": "PATIENT INFORMATION\nMedical Record #: MRN-12345678\nName: John Smith\nSSN: 123-45-6789\nEmail: jsmith@email.net\nCredit Card: 4532-1234-5678-9012\nPhone: (555) 123-4567\nAPI Key: ak_live_HJd8e7h23hFxMznWcQE5TWqL",
  "source": "healthcare_system",
  "classification": "confidential"
}
```

## Pipeline Architecture

The pipeline we will be using in this example is a hybrid feed-forward pipeline where data flows through both regex and AI processing stages. The pipeline combines fast regex pattern matching with semantic analysis for optimal performance.

Below is a visualization of the pipeline showing all stages and data flow:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│ DLPInputProcessor│───▶│ RegexProcessor  │
│   Documents     │    │ (Preprocessing) │    │  (Fast Filter)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RiskScorer    │◀───│ DLPPipeline     │◀───│ GliNERProcessor │
│ (Risk Analysis) │    │ (Orchestration) │    │ (AI Validation) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **DLPInputProcessor** | Text preprocessing & chunking |
| **RegexProcessor** | Fast pattern matching 
| **GliNERProcessor** | Semantic analysis | 
| **RiskScorer** | Threat assessment |
| **DLPPipeline** | Orchestration & metrics | 

## Setup

This example utilizes the Triton Inference Server to perform AI inference for semantic analysis. The GLiNER model is used for contextual entity validation.

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, for accelerated AI processing)

#### 1. Clone the Repository



#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Download SLM model
```bash
# The GLiNER model will be downloaded automatically on first use
python -c "from gliner import GLiNER; GLiNER.from_pretrained('gretelai/gretel-gliner-bi-small-v1.0')"
```

## Running the Pipeline

With the Morpheus CLI, the entire DLP pipeline can be configured and run. Using the `morpheus run pipeline-dlp` command, we can build the pipeline by specifying each stage's configuration.

### Quick Start Example

```python
from dlp_workflow import DLPPipeline

# Load regex patterns
import json
with open('regex_patterns.json', 'r') as f:
    patterns = json.load(f)

# Initialize the hybrid DLP pipeline
dlp_pipeline = DLPPipeline(
    regex_patterns=patterns,
    confidence_threshold=0.7,
    model_name="gretelai/gretel-gliner-bi-small-v1.0",
    context_window=300
)

# Process a document
document = """
PATIENT INFORMATION
Medical Record #: MRN-12345678
Name: John Smith
SSN: 123-45-6789
Email: jsmith@email.net
Credit Card: 4532-1234-5678-9012
"""

# Full pipeline processing
results = dlp_pipeline.process(document)

print(f"Risk Score: {results['risk_assessment']['risk_score']}/100")
print(f"Risk Level: {results['risk_assessment']['risk_level']}")
print(f"Total Findings: {results['total_findings']}")
print(f"Processing Time: {results['performance_metrics']['total_time']:.3f}s")
```

### Command Line Interface


##  License


## Acknowledgments

- **GLiNER Team**: Excellent semantic analysis model
- **Gretel.ai**: High-quality synthetic datasets and finetuned GLiNER model for PII
- **nervaluate**: Comprehensive NER evaluation metrics
