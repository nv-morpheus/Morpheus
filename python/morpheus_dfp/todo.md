<!--
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Tasks
- [X] Refactor Morpheus production stages

- [X] Add morpheus_dfp Conda recipe

- [X] Update CI to build and upload morpheus_dfp Conda package

- [] Update docs to include the DFP apis and README for using the DFP library

- [] Move DFP unit tests from tests/examples/digital_fingerprinting to tests/morpheus_dfp

- [] Refactor DFP benchmarks

- [] Update DFP example docker file to install the morpheus_dfp package instead of using the Morpheus image as base

- [] Consolidate version file used in setup.py across all Morpheus packages


# Q&A for future reference
1. Do we refactor sample pipelines to python/morpheus_dfp/morpheus_dfp/pipeline?
  No. They are not part of the library. They are just examples.

2. Do we refactor data (just the script for pulling the data, fetch_example_data.py) used for running the sample DFP pipelines?
   No. Same as above.

3. Do we refactor Morpheus DFP starter example?
   No. Starter will be dropped, #1715

4. Visualizations?
   No. Sample pipeline.

5. Demo?
   No. Sample pipeline.

6. Refactor notebooks?
   No. Sample only.

7. Refactor DFP example control messages?
   No.
