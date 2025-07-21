<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Known Issues

- `ransomware_detection` example pipeline occasionally logs a `distributed.comm.core.CommClosedError` error during shutdown ([#2026](https://github.com/nv-morpheus/Morpheus/issues/2026)).
- Arm64 users need to install CUDA enabled PyTorch by hand ([#2095](https://github.com/nv-morpheus/Morpheus/issues/2095))
- Performance issues were observed running the `abp_pcap_detection` and `ransomware_detection` pipelines on AArch64 ([#2120](https://github.com/nv-morpheus/Morpheus/issues/2120)) & ([#2124](https://github.com/nv-morpheus/Morpheus/issues/2124)) on Ubuntu 22.04. Arm64 users should consider upgrading to Ubuntu 24.04.
- LLM `vdb_upload` and `rag` pipelines not supported on AArch64 ([#2122](https://github.com/nv-morpheus/Morpheus/issues/2122))
- `gnn_fraud_detection_pipeline` not working on AArch64 ([#2123](https://github.com/nv-morpheus/Morpheus/issues/2123))
- DFP visualization fails to install on AArch64 ([#2125](https://github.com/nv-morpheus/Morpheus/issues/2125))
- Build errors after a CMake reconfigure, work-around is to delete the `build` directory and re-run CMake ([#2279](https://github.com/nv-morpheus/Morpheus/issues/2279)).
- C++ examples fail to build inside the release container on AArch64 ([#2276](https://github.com/nv-morpheus/Morpheus/issues/2276)).
- DOCA Builds failing inside devcontainer ([#2285](https://github.com/nv-morpheus/Morpheus/issues/2285))
- Python stubgen failing for C++ examples inside devcontainer ([#2287](https://github.com/nv-morpheus/Morpheus/issues/2287)), the actual build succeeds, and code is functional, but the Python stub files are not generated.


Refer to [open issues in the Morpheus project](https://github.com/nv-morpheus/Morpheus/issues)
