# Data Loss Prevention (DLP) Pipeline with Morpheus

This example demonstrates how to use Morpheus to implement a DLP solution that combines regex pattern matching with small language model (SLM) powered semantic analysis for data protection.

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |
| Morpheus Release Container | ✔ |  |
| Dev Container | ✔ |  |


## Background

The goal of this  DLP pipeline is to identify and mitigate exposure of sensitive information in enterprise data streams as quickly and accurately as possible. Effective DLP systems are essential in today's enterprise and cloud environments to:

- **Prevent Sensitive Data Leakage**: Identify and mitigate exposure of personally identifiable information (PII), protected health information (PHI), credentials, and intellectual property
- **Ensure Regulatory Compliance**: Meet data protection mandates such as GDPR, HIPAA, PCI-DSS, and ISO 27001
- **Detect Insider Threats**: Monitor and flag suspicious access to sensitive data by internal actors
- **Secure Cloud and SaaS Workloads**: Protect data in motion and at rest across hybrid and multi-cloud architectures

This pipeline implements a hybrid approach that couples:
- **Regex-based pre-filtering** for fast and scalable detection with high recall 
- **Contextual entity validation** using GLiNER-based semantic analysis

## 🎯 Supported Data Types

The DLP pipeline is capable of detecting multiple categories of sensitive information including these and more entities:

### Personal Information
1. **SSN**: Social Security Numbers with flexible formatting
2. **Credit Cards**: Visa, MasterCard, American Express and Discover with validation
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
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Text    │───▶│ DLPInputProcessor│───▶│ RegexProcessor  │
│   Documents     │     │ (Preprocessing)  │     │  (Fast Filter)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐                              ┌─────────────────┐
│   RiskScorer    │◀────────────────────────────│ GliNERProcessor │
│ (Risk Analysis) │                              │ (AI Validation) │
└─────────────────┘                              └─────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **DLPInputProcessor** | Text preprocessing & chunking |
| **RegexProcessor** | Fast pattern matching
| **GliNERProcessor** | Semantic analysis |
| **RiskScorer** | Threat assessment |


## Running the Pipeline

From the root of the Morpheus repo, run:
```bash
python examples/data_loss_prevention/run.py --help
```

Output:
```
Usage: run.py [OPTIONS]

Options:
  --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                  Specify the logging level to use.  [default:
                                  INFO]
  --regex_file PATH               JSON file containing regex patterns
                                  [default: /home/dagardner/work/morpheus/exam
                                  ples/data_loss_prevention/data/regex_pattern
                                  s.json]
  --dataset TEXT                  Specify the datasets to use, can be set
                                  multiple times, valid datasets are: gretel.
                                  [default: gretel]
  --num_samples INTEGER           Number of samples to use from each dataset,
                                  set to -1 for all samples.  [default: 2000]
  --out_file FILE                 Output file  [required]
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the default arguments, run the following:

```bash
python examples/data_loss_prevention/run.py
```
