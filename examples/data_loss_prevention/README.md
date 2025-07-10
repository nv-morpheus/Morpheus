<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Data Loss Prevention (DLP) Pipeline with Morpheus

This example demonstrates how to use Morpheus to implement a high-performance DLP solution for sensitive data detection. It combines regex pattern matching with semantic analysis powered by a small language model (SLM), enabling efficient and accurate identification of sensitive information in unstructured text.

## Supported Environments
All environments require additional Conda packages which can be installed with either the `conda/environments/all_cuda-128_arch-$(arch).yaml` environment file. Refer to the [Setup](#setup) section for more information.
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |
| Morpheus Release Container | ✔ |  |
| Dev Container | ✔ |  |

### Supported Architectures
| Architecture | Supported | Issue |
|--------------|-----------|-------|
| x86_64 | ✔ | |
| aarch64 | ✘ | Work-around available refer to: [#2095](https://github.com/nv-morpheus/Morpheus/issues/2095) |

## Background

The goal of this  DLP pipeline is to identify and mitigate exposure of sensitive information in enterprise data streams as quickly and accurately as possible. Effective DLP systems are essential in today's enterprise and cloud environments to:

- **Prevent Sensitive Data Leakage**: Identify and mitigate exposure of personally identifiable information (PII), protected health information (PHI), credentials, and intellectual property
- **Ensure Regulatory Compliance**: Meet data protection mandates such as GDPR, HIPAA, PCI-DSS, and ISO 27001
- **Detect Insider Threats**: Monitor and flag suspicious access to sensitive data by internal actors
- **Secure Cloud and SaaS Workloads**: Protect data in motion and at rest across hybrid and multi-cloud architectures

This pipeline implements a hybrid approach that couples:
- **Regex-based pre-filtering** for fast and scalable detection with high recall
- **Contextual entity validation** using GLiNER-based semantic analysis

##  Supported Data Types

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

## Dataset

The dataset that this workflow processes can be various text formats including JSON, CSV, or plain text documents converted into *textual data* of CSV format.
For this pipeline example, the input is a CSV with `source_text` column indicating the target text. For example, below is a sample document containing multiple types of sensitive information indicated in the `source_text` field:

```json
{
  "timestamp": 1616380971990,
  "document_type": "patient_record",
  "source_text": "PATIENT INFORMATION\nMedical Record #: MRN-12345678\nName: John Smith\nSSN: 123-45-6789\nEmail: jsmith@email.net\nCredit Card: 4532-1234-5678-9012\nPhone: (555) 123-4567\nAPI Key: ak_live_HJd8e7h23hFxMznWcQE5TWqL",
  "source": "healthcare_system",
  "classification": "confidential"
}
```

## Pipeline Architecture

The pipeline we will be using in this example is a hybrid feed-forward pipeline where data flows through both regex and AI processing stages. The pipeline combines fast regex pattern matching with semantic analysis for optimal performance.

Below is a visualization of the pipeline showing all stages and data flow:

```
┌────────────────────┐
│    Input Text      │
│    Documents       │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  DLPInputProcessor │-▶│   RegexProcessor   │-▶│   GliNERProcessor  │-▶│     RiskScorer     │
│   (Preprocessing)  │   │    (Fast Filter)   │   │   (AI Validation)  │   │   (Risk Analysis)  │
└────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘

```

### Core Components

| Component | Purpose |
|-----------|---------|
| **DLPInputProcessor** | Text preprocessing & chunking |
| **RegexProcessor** | Fast pattern matching
| **GliNERProcessor** | Semantic analysis |
| **RiskScorer** | Threat assessment |

## Setup

### Installing GliNER and dependencies

Prior to running the DLP pipeline, additional requirements must be installed in to your Conda environment.

```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/all_cuda-128_arch-$(arch).yaml
```

Install the GliNER package itself: <!-- Work-around for https://github.com/urchade/GLiNER/issues/267 -->
```bash
pip install --no-deps gliner==0.2.21
```

### Downloading the model

The model is stored in the Morpheus repository using [Git Large File Storage (LFS)](https://git-lfs.github.com/). Only those files which are strictly needed to run Morpheus are downloaded by default when the repository is cloned. The model can be downloaded using the `fetch_data.py` script.

```bash
git lfs install
./scripts/fetch_data.py fetch models
```

### Build the Example

#### From Source
For users who cloned the Morpheus repository can build the example by passing in the `-DMORPHEUS_BUILD_EXAMPLES=ON` flag to CMake, for users using the `scripts/compile.sh` at the root of the Morpheus repo can do this by setting the `CMAKE_CONFIGURE_EXTRA_ARGS` environment variable:
```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_EXAMPLES=ON" ./scripts/compile.sh
```

#### From the Docker container
For users using the Morpheus Docker container, the example can be built by running the following:
```bash
cd /workspace/examples/data_loss_prevention
./compile.sh

pip install ./
```

### Launching Triton

This example utilizes the Triton Inference Server to perform inference. Pull the Docker image for Triton:
```bash
docker pull nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.06
```

Run the following to launch Triton and load the `gliner-bi-encoder-onnx` model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.06 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model gliner-bi-encoder-onnx
```

This will launch Triton and only load the `gliner-bi-encoder-onnx` model.

Once Triton has loaded the model, the following will be displayed:

```
+------------------------+---------+--------+
| Model                  | Version | Status |
+------------------------+---------+--------+
| gliner-bi-encoder-onnx | 1       | READY  |
+------------------------+---------+--------+
```

> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.


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
                                  [default: examples/data_loss_prevention/data/regex_patterns.json]
  --dataset TEXT                  Specify the datasets to use, can be set
                                  multiple times, valid datasets are: gretel.
                                  [default: gretel]
  --input_file FILE               Input file to use, if specified, overrides
                                  the dataset option.
  --include_privacy_masks         Include privacy masks in the output
                                  DataFrame, ignored if --input_file is set.
                                  This is useful for evaluation.
  --num_samples INTEGER           Number of samples to use from each dataset,
                                  ignored if --input_file is set, set to -1
                                  for all samples.  [default: 2000]
  --repeat INTEGER                Repeat the input dataset, useful for
                                  testing. A value of 1 means no repeat.
                                  [default: 1]
  --regex_only                    Only perform regex matching and skip the
                                  GliNER processor.
  --server_url TEXT               Tritonserver url.  [default: localhost:8001;
                                  required]
  --model_max_batch_size INTEGER  Maximum batch size for model inference, used
                                  by the GliNER processor. Larger values may
                                  improve performance but require more GPU
                                  memory.  [default: 16]
  --model_source_dir DIRECTORY    Directory containing the GliNER model files
                                  [default: models/dlp_models/gliner_bi_encoder]
  --out_file FILE                 Output file  [default: .tmp/output/data_loss_prevention.jsonlines;
                                  required]
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the default arguments, run the following:

```bash
python examples/data_loss_prevention/run.py
```
