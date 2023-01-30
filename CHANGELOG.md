<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Morpheus 22.11.00 (18 Nov 2022)

## 🐛 Bug Fixes

- Set ver of mlflow client to match that of the server ([#484](https://github.com/nv-morpheus/Morpheus/pull/484)) [@dagardner-nv](https://github.com/dagardner-nv)
- Improve Morpheus Shutdown Behavior On Exception ([#478](https://github.com/nv-morpheus/Morpheus/pull/478)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- update for numpy array results on ABP notebook ([#468](https://github.com/nv-morpheus/Morpheus/pull/468)) [@gbatmaz](https://github.com/gbatmaz)
- Remove warning about tests not having a return value ([#457](https://github.com/nv-morpheus/Morpheus/pull/457)) [@dagardner-nv](https://github.com/dagardner-nv)
- DFPFileBatcherStage: Sort only by timestamp ([#450](https://github.com/nv-morpheus/Morpheus/pull/450)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix unittests ([#444](https://github.com/nv-morpheus/Morpheus/pull/444)) [@dagardner-nv](https://github.com/dagardner-nv)
- Ensure Camouflage is shutdown after every test ([#436](https://github.com/nv-morpheus/Morpheus/pull/436)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Properly calculate the offset of the column_view &amp; apply offsets in copy_meta_ranges ([#423](https://github.com/nv-morpheus/Morpheus/pull/423)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove buildx from docker command causes issues with docker 20.10.5 ([#417](https://github.com/nv-morpheus/Morpheus/pull/417)) [@dagardner-nv](https://github.com/dagardner-nv)
- Pin camouflage to v0.9 and ensure pytest-benchmark&gt;=4 ([#416](https://github.com/nv-morpheus/Morpheus/pull/416)) [@dagardner-nv](https://github.com/dagardner-nv)
- Set a default value for --columns_file and populate the help string ([#405](https://github.com/nv-morpheus/Morpheus/pull/405)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add missing import ([#402](https://github.com/nv-morpheus/Morpheus/pull/402)) [@dagardner-nv](https://github.com/dagardner-nv)
- Move to srf-22.11 alpha ([#399](https://github.com/nv-morpheus/Morpheus/pull/399)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Use tritonserver 22.06 for the phishing example ([#477](https://github.com/nv-morpheus/Morpheus/pull/477)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add datasets readme, change lfs tracking ([#474](https://github.com/nv-morpheus/Morpheus/pull/474)) [@raykallen](https://github.com/raykallen)
- Documentation Updates ([#458](https://github.com/nv-morpheus/Morpheus/pull/458)) [@bsuryadevara](https://github.com/bsuryadevara)
- Set channels and versions to ensure we get a good version of tensorflow ([#429](https://github.com/nv-morpheus/Morpheus/pull/429)) [@dagardner-nv](https://github.com/dagardner-nv)
- Clarifications and improvements to Kafka manual testing documentation ([#422](https://github.com/nv-morpheus/Morpheus/pull/422)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add documentation to LinearBoundary stages ([#418](https://github.com/nv-morpheus/Morpheus/pull/418)) [@drobison00](https://github.com/drobison00)
- Update Quickstart Guide ([#398](https://github.com/nv-morpheus/Morpheus/pull/398)) [@bsuryadevara](https://github.com/bsuryadevara)
- Add more details to E2E benchmarks README ([#395](https://github.com/nv-morpheus/Morpheus/pull/395)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🚀 New Features

- Phishing model and data updates ([#462](https://github.com/nv-morpheus/Morpheus/pull/462)) [@raykallen](https://github.com/raykallen)
- Root cause analysis example pipeline ([#460](https://github.com/nv-morpheus/Morpheus/pull/460)) [@efajardo-nv](https://github.com/efajardo-nv)
- Root cause use case scripts ([#452](https://github.com/nv-morpheus/Morpheus/pull/452)) [@gbatmaz](https://github.com/gbatmaz)
- DFP Visualization Example ([#439](https://github.com/nv-morpheus/Morpheus/pull/439)) [@efajardo-nv](https://github.com/efajardo-nv)
- : Replacing md issue templates with yml forms ([#407](https://github.com/nv-morpheus/Morpheus/pull/407)) [@jarmak-nv](https://github.com/jarmak-nv)
- Create action to add issues/prs to the project ([#326](https://github.com/nv-morpheus/Morpheus/pull/326)) [@jarmak-nv](https://github.com/jarmak-nv)

## 🛠️ Improvements

- Update External Repo to Single Directory ([#479](https://github.com/nv-morpheus/Morpheus/pull/479)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Root cause analysis datasets ([#470](https://github.com/nv-morpheus/Morpheus/pull/470)) [@efajardo-nv](https://github.com/efajardo-nv)
- Various Dockerfile-based updates for 22.11 ([#466](https://github.com/nv-morpheus/Morpheus/pull/466)) [@pdmack](https://github.com/pdmack)
- ABP change the arg name in the comment and update req ([#455](https://github.com/nv-morpheus/Morpheus/pull/455)) [@gbatmaz](https://github.com/gbatmaz)
- Forward Merge `branch-22.09` into `branch-22.09` ([#448](https://github.com/nv-morpheus/Morpheus/pull/448)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Avoid unnecessary copy in add-scores stage ([#438](https://github.com/nv-morpheus/Morpheus/pull/438)) [@dagardner-nv](https://github.com/dagardner-nv)
- MultiInferenceMessage &amp; MultiResponseMessage share a new base class ([#419](https://github.com/nv-morpheus/Morpheus/pull/419)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add dockerfile for CI runner ([#408](https://github.com/nv-morpheus/Morpheus/pull/408)) [@dagardner-nv](https://github.com/dagardner-nv)
- remove work-around for pytest-kafka issue #10 ([#392](https://github.com/nv-morpheus/Morpheus/pull/392)) [@dagardner-nv](https://github.com/dagardner-nv)
- Switch to Github Actions ([#389](https://github.com/nv-morpheus/Morpheus/pull/389)) [@dagardner-nv](https://github.com/dagardner-nv)
- Simplify Python impl for KafkaSourceStage ([#300](https://github.com/nv-morpheus/Morpheus/pull/300)) [@dagardner-nv](https://github.com/dagardner-nv)

# Morpheus 22.09.01 (9 Nov 2022)

## 🐛 Bug Fixes

- 426 bug msg keyerror data ([#428](https://github.com/nv-morpheus/Morpheus/pull/428)) [@bsuryadevara](https://github.com/bsuryadevara)

# Morpheus 22.09.00 (30 Sep 2022)

## 🚨 Breaking Changes

- Adjustments for docker build inclusion of morpheus-visualization ([#366](https://github.com/nv-morpheus/Morpheus/pull/366)) [@pdmack](https://github.com/pdmack)
- Add pluggy to docker build for CLI; fixes 357 ([#358](https://github.com/nv-morpheus/Morpheus/pull/358)) [@pdmack](https://github.com/pdmack)
- kafka integration tests ([#308](https://github.com/nv-morpheus/Morpheus/pull/308)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🐛 Bug Fixes

- Pin numba version ([#387](https://github.com/nv-morpheus/Morpheus/pull/387)) [@pdmack](https://github.com/pdmack)
- Training and validation script and notebook fixes. ([#386](https://github.com/nv-morpheus/Morpheus/pull/386)) [@shawn-davis](https://github.com/shawn-davis)
- Add websockets dependency for viz ([#383](https://github.com/nv-morpheus/Morpheus/pull/383)) [@efajardo-nv](https://github.com/efajardo-nv)
- updating the filenames in the paths and adding the missing file for abp ([#378](https://github.com/nv-morpheus/Morpheus/pull/378)) [@gbatmaz](https://github.com/gbatmaz)
- Ransomware models need update for Triton 22.08 ([#377](https://github.com/nv-morpheus/Morpheus/pull/377)) [@bsuryadevara](https://github.com/bsuryadevara)
- Adjustments for docker build inclusion of morpheus-visualization ([#366](https://github.com/nv-morpheus/Morpheus/pull/366)) [@pdmack](https://github.com/pdmack)
- Removing `grpcio-channelz` ([#360](https://github.com/nv-morpheus/Morpheus/pull/360)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add pluggy to docker build for CLI; fixes 357 ([#358](https://github.com/nv-morpheus/Morpheus/pull/358)) [@pdmack](https://github.com/pdmack)
- Branch 22.09 merge 22.08 ([#336](https://github.com/nv-morpheus/Morpheus/pull/336)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 📖 Documentation

- Update Quickstart Guide for 22.09 ([#382](https://github.com/nv-morpheus/Morpheus/pull/382)) [@pdmack](https://github.com/pdmack)
- Starter DFP docstring updates ([#370](https://github.com/nv-morpheus/Morpheus/pull/370)) [@efajardo-nv](https://github.com/efajardo-nv)
- Wholesale updates for tritonserver version ([#369](https://github.com/nv-morpheus/Morpheus/pull/369)) [@pdmack](https://github.com/pdmack)
- Misc DFP Documentation &amp; fixes ([#368](https://github.com/nv-morpheus/Morpheus/pull/368)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix broken developer guide code examples ([#333](https://github.com/nv-morpheus/Morpheus/pull/333)) [@dagardner-nv](https://github.com/dagardner-nv)
- Clarification of known issue in QSG ([#328](https://github.com/nv-morpheus/Morpheus/pull/328)) [@pdmack](https://github.com/pdmack)
- Initial example readme ([#304](https://github.com/nv-morpheus/Morpheus/pull/304)) [@cwharris](https://github.com/cwharris)

## 🚀 New Features

- Bump Versions 22.09 ([#361](https://github.com/nv-morpheus/Morpheus/pull/361)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Demo notebook for DFP feature selection with toy dataset ([#359](https://github.com/nv-morpheus/Morpheus/pull/359)) [@hsin-c](https://github.com/hsin-c)
- SID Visualization Example ([#354](https://github.com/nv-morpheus/Morpheus/pull/354)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add DFP Production Example Workflow ([#352](https://github.com/nv-morpheus/Morpheus/pull/352)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Digital Fingerprinting 2.0 Pipelines ([#315](https://github.com/nv-morpheus/Morpheus/pull/315)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add CLI Extensions to Allow Running Custom Stages with the Morpheus CLI ([#312](https://github.com/nv-morpheus/Morpheus/pull/312)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🛠️ Improvements

- Update HAMMAH E2E benchmark test to DFP 2.0 ([#380](https://github.com/nv-morpheus/Morpheus/pull/380)) [@efajardo-nv](https://github.com/efajardo-nv)
- Resolves #350 -- Adds support for segmenting LinearPipelines ([#374](https://github.com/nv-morpheus/Morpheus/pull/374)) [@drobison00](https://github.com/drobison00)
- Add jupyter to the main Dockerfile (NGC) ([#373](https://github.com/nv-morpheus/Morpheus/pull/373)) [@pdmack](https://github.com/pdmack)
- Use libcudacxx from NVIDIA repo ([#372](https://github.com/nv-morpheus/Morpheus/pull/372)) [@pdmack](https://github.com/pdmack)
- DFP Updates to enable example workflows ([#371](https://github.com/nv-morpheus/Morpheus/pull/371)) [@drobison00](https://github.com/drobison00)
- Mlflow update 2209 ([#367](https://github.com/nv-morpheus/Morpheus/pull/367)) [@pdmack](https://github.com/pdmack)
- Update Dockerfile ([#353](https://github.com/nv-morpheus/Morpheus/pull/353)) [@pdmack](https://github.com/pdmack)
- kafka integration tests ([#308](https://github.com/nv-morpheus/Morpheus/pull/308)) [@dagardner-nv](https://github.com/dagardner-nv)
- IWYU CI integration for Morpheus ([#287](https://github.com/nv-morpheus/Morpheus/pull/287)) [@dagardner-nv](https://github.com/dagardner-nv)

## Known Issues

- The GNN example workflow cannot be run due to incompatible dependencies between cuML 22.08 and SRF 22.09. See issue [#390](https://github.com/nv-morpheus/Morpheus/issues/390)

# Morpheus 22.08.00 (7 Sep 2022)

## 🐛 Bug Fixes

- Fixing compilation with SRF 22.08a ([#332](https://github.com/nv-morpheus/Morpheus/pull/332)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update Ransomware Models ([#295](https://github.com/nv-morpheus/Morpheus/pull/295)) [@bsuryadevara](https://github.com/bsuryadevara)
- Bump numpy from 1.19.5 to 1.22.0 in /models/validation-inference-scripts/fraud-detection-models ([#175](https://github.com/nv-morpheus/Morpheus/pull/175)) @dependabot[bot]
- Bump numpy from 1.19.5 to 1.22.0 in /models/training-tuning-scripts/fraud-detection-models ([#174](https://github.com/nv-morpheus/Morpheus/pull/174)) @dependabot[bot]
- Bump numpy from 1.20.3 to 1.22.0 in /models/validation-inference-scripts/phishing-models ([#173](https://github.com/nv-morpheus/Morpheus/pull/173)) @dependabot[bot]
- Bump numpy from 1.20.3 to 1.22.0 in /models/validation-inference-scripts/hammah-models ([#172](https://github.com/nv-morpheus/Morpheus/pull/172)) @dependabot[bot]

## 📖 Documentation

- Add Azure known issue to QSG ([#323](https://github.com/nv-morpheus/Morpheus/pull/323)) [@pdmack](https://github.com/pdmack)
- QSG updates for data dir ([#302](https://github.com/nv-morpheus/Morpheus/pull/302)) [@pdmack](https://github.com/pdmack)
- Manual testing of Morpheus with Kafka &amp; Validation improvements ([#290](https://github.com/nv-morpheus/Morpheus/pull/290)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updates README.md to include instructions for launching Triton ([#289](https://github.com/nv-morpheus/Morpheus/pull/289)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update NLP SI example readme ([#284](https://github.com/nv-morpheus/Morpheus/pull/284)) [@pdmack](https://github.com/pdmack)
- Update GNN example readme ([#283](https://github.com/nv-morpheus/Morpheus/pull/283)) [@pdmack](https://github.com/pdmack)

## 🚀 New Features

- E2E benchmark tests ([#269](https://github.com/nv-morpheus/Morpheus/pull/269)) [@efajardo-nv](https://github.com/efajardo-nv)
- Use SRF Logging ([#266](https://github.com/nv-morpheus/Morpheus/pull/266)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🛠️ Improvements

- Fixes issues with NLP pipelines when data is not truncated ([#316](https://github.com/nv-morpheus/Morpheus/pull/316)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update include ordering and style ([#277](https://github.com/nv-morpheus/Morpheus/pull/277)) [@dagardner-nv](https://github.com/dagardner-nv)
- MonitorStage add a blank space between the rate and the unit description ([#275](https://github.com/nv-morpheus/Morpheus/pull/275)) [@dagardner-nv](https://github.com/dagardner-nv)
- Ensure TableInfo only appears in one lib ([#273](https://github.com/nv-morpheus/Morpheus/pull/273)) [@dagardner-nv](https://github.com/dagardner-nv)
- copy multiple ranges for MultiMessage ([#231](https://github.com/nv-morpheus/Morpheus/pull/231)) [@dagardner-nv](https://github.com/dagardner-nv)
- GPU not needed for build &amp; documentation CI stages ([#181](https://github.com/nv-morpheus/Morpheus/pull/181)) [@dagardner-nv](https://github.com/dagardner-nv)


# Morpheus 22.06.01 (15 Jul 2022)

## 🐛 Bug Fixes

- Fix pandas version in runtime container ([#270](https://github.com/nv-morpheus/Morpheus/pull/270)) [@efajardo-nv](https://github.com/efajardo-nv)


# Morpheus 22.06.00 (5 Jul 2022)

## 🚨 Breaking Changes

- Update Morpheus to Use SRF 22.06 ([#152](https://github.com/nv-morpheus/Morpheus/pull/152)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- update cudf to 22.04, cuda to 11.5 ([#148](https://github.com/nv-morpheus/Morpheus/pull/148)) [@cwharris](https://github.com/cwharris)
- Fixes Timestamp Nodes When Running with `--debug` ([#145](https://github.com/nv-morpheus/Morpheus/pull/145)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Dockerfile COPY section needs update ([#141](https://github.com/nv-morpheus/Morpheus/pull/141)) [@pdmack](https://github.com/pdmack)
- Reorganize the python package files ([#98](https://github.com/nv-morpheus/Morpheus/pull/98)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 🐛 Bug Fixes

- Fixing Python Kafka Source with Multiple Threads ([#262](https://github.com/nv-morpheus/Morpheus/pull/262)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixing the from-kafka stage ([#257](https://github.com/nv-morpheus/Morpheus/pull/257)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Pin Conda Package `cuda-python <=11.7.0` to Fix Conda Build ([#252](https://github.com/nv-morpheus/Morpheus/pull/252)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixing `to-kafka` Stage by Converting to a Pass Through Node ([#245](https://github.com/nv-morpheus/Morpheus/pull/245)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Move Morpheus Data Files Out of LFS ([#242](https://github.com/nv-morpheus/Morpheus/pull/242)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Missing 'supports_cpp_node' impl function ([#239](https://github.com/nv-morpheus/Morpheus/pull/239)) [@bsuryadevara](https://github.com/bsuryadevara)
- Revert ransomware feature config changes ([#234](https://github.com/nv-morpheus/Morpheus/pull/234)) [@bsuryadevara](https://github.com/bsuryadevara)
- Use git version 2.35.3 in release build ([#224](https://github.com/nv-morpheus/Morpheus/pull/224)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Rollback CI images and fix dev container ([#191](https://github.com/nv-morpheus/Morpheus/pull/191)) [@cwharris](https://github.com/cwharris)
- Update cuda11.5_dev.yml ([#167](https://github.com/nv-morpheus/Morpheus/pull/167)) [@pdmack](https://github.com/pdmack)
- Adding in pybind11-stubgen to the conda package ([#163](https://github.com/nv-morpheus/Morpheus/pull/163)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixes Timestamp Nodes When Running with `--debug` ([#145](https://github.com/nv-morpheus/Morpheus/pull/145)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Dockerfile COPY section needs update ([#141](https://github.com/nv-morpheus/Morpheus/pull/141)) [@pdmack](https://github.com/pdmack)
- Add pybind11-stubgen to conda environment yaml. ([#109](https://github.com/nv-morpheus/Morpheus/pull/109)) [@drobison00](https://github.com/drobison00)
- Update hammah-inference.py ([#90](https://github.com/nv-morpheus/Morpheus/pull/90)) [@pdmack](https://github.com/pdmack)
- Install new apt key in docker ([#72](https://github.com/nv-morpheus/Morpheus/pull/72)) [@dagardner-nv](https://github.com/dagardner-nv)
- Ensure default path values in morpheus.cli are no longer relative ([#62](https://github.com/nv-morpheus/Morpheus/pull/62)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix the nlp_si_detection example ([#61](https://github.com/nv-morpheus/Morpheus/pull/61)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Fix GNN example and update installation instructions ([#189](https://github.com/nv-morpheus/Morpheus/pull/189)) [@cwharris](https://github.com/cwharris)
- Documentation fixes ([#147](https://github.com/nv-morpheus/Morpheus/pull/147)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update morpheus_quickstart_guide.md ([#142](https://github.com/nv-morpheus/Morpheus/pull/142)) [@pdmack](https://github.com/pdmack)
- Split data dir, moving large files into examples/data ([#130](https://github.com/nv-morpheus/Morpheus/pull/130)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update morpheus_quickstart_guide.md ([#127](https://github.com/nv-morpheus/Morpheus/pull/127)) [@bsuryadevara](https://github.com/bsuryadevara)
- Update morpheus_quickstart_guide.md ([#106](https://github.com/nv-morpheus/Morpheus/pull/106)) [@pdmack](https://github.com/pdmack)
- Apply enterprise sphinx html theme to docs ([#97](https://github.com/nv-morpheus/Morpheus/pull/97)) [@efajardo-nv](https://github.com/efajardo-nv)
- Updates to developer_guide for clarity. ([#96](https://github.com/nv-morpheus/Morpheus/pull/96)) [@lobotmcj](https://github.com/lobotmcj)
- Updates to README.md for clarity ([#91](https://github.com/nv-morpheus/Morpheus/pull/91)) [@BartleyR](https://github.com/BartleyR)
- Update README.md ([#76](https://github.com/nv-morpheus/Morpheus/pull/76)) [@pdmack](https://github.com/pdmack)
- Fix typos in README.md & Change GitLab-style reference in CONTRIBUTING.md ([#74](https://github.com/nv-morpheus/Morpheus/pull/74)) [@lobotmcj](https://github.com/lobotmcj)
- Updated README with documentation links and banner image ([#71](https://github.com/nv-morpheus/Morpheus/pull/71)) [@BartleyR](https://github.com/BartleyR)

## 🚀 New Features

- Fix 22.06 style checks ([#249](https://github.com/nv-morpheus/Morpheus/pull/249)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add CLI Relative Path Fallback ([#232](https://github.com/nv-morpheus/Morpheus/pull/232)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Adding new SRF cmake variables ([#198](https://github.com/nv-morpheus/Morpheus/pull/198)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- ransomware ds requirements ([#196](https://github.com/nv-morpheus/Morpheus/pull/196)) [@raykallen](https://github.com/raykallen)
- Update to rapids 22.06 ([#180](https://github.com/nv-morpheus/Morpheus/pull/180)) [@cwharris](https://github.com/cwharris)
- Fix for CI check script ([#158](https://github.com/nv-morpheus/Morpheus/pull/158)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update Morpheus to Use SRF 22.06 ([#152](https://github.com/nv-morpheus/Morpheus/pull/152)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Ransomware detection pipeline ([#149](https://github.com/nv-morpheus/Morpheus/pull/149)) [@bsuryadevara](https://github.com/bsuryadevara)
- update cudf to 22.04, cuda to 11.5 ([#148](https://github.com/nv-morpheus/Morpheus/pull/148)) [@cwharris](https://github.com/cwharris)
- Limit which lfs assets are pulled by default ([#139](https://github.com/nv-morpheus/Morpheus/pull/139)) [@dagardner-nv](https://github.com/dagardner-nv)
- AppShieldSource stage ([#136](https://github.com/nv-morpheus/Morpheus/pull/136)) [@bsuryadevara](https://github.com/bsuryadevara)
- Include C++ Unittests in CI ([#135](https://github.com/nv-morpheus/Morpheus/pull/135)) [@dagardner-nv](https://github.com/dagardner-nv)
- Set Python3_FIND_STRATEGY=Location ([#131](https://github.com/nv-morpheus/Morpheus/pull/131)) [@dagardner-nv](https://github.com/dagardner-nv)
- Migrate Neo's tensor code directly into Morpheus ([#129](https://github.com/nv-morpheus/Morpheus/pull/129)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updating CODEOWNERS for New Organization ([#118](https://github.com/nv-morpheus/Morpheus/pull/118)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add pybind11/cython stubs to Morpheus package ([#100](https://github.com/nv-morpheus/Morpheus/pull/100)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Reorganize the python package files ([#98](https://github.com/nv-morpheus/Morpheus/pull/98)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update morpheus dev container to support flag for building with debug python build + source files. ([#81](https://github.com/nv-morpheus/Morpheus/pull/81)) [@drobison00](https://github.com/drobison00)

## 🛠️ Improvements

- Fixing Outstanding Style Errors ([#261](https://github.com/nv-morpheus/Morpheus/pull/261)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Pin cuda-python to 11.7.0 ([#246](https://github.com/nv-morpheus/Morpheus/pull/246)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fixes Pipeline.visualize ([#203](https://github.com/nv-morpheus/Morpheus/pull/203)) [@dagardner-nv](https://github.com/dagardner-nv)
- nlp_si_detection example improvements ([#193](https://github.com/nv-morpheus/Morpheus/pull/193)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update  GNN FSI pipeline example & notebook. ([#182](https://github.com/nv-morpheus/Morpheus/pull/182)) [@tzemicheal](https://github.com/tzemicheal)
- Add missing git-lfs package to docker container ([#179](https://github.com/nv-morpheus/Morpheus/pull/179)) [@dagardner-nv](https://github.com/dagardner-nv)
- release container build fixes ([#164](https://github.com/nv-morpheus/Morpheus/pull/164)) [@cwharris](https://github.com/cwharris)
- remove ssh instructions from CONTRIBUTING guide ([#162](https://github.com/nv-morpheus/Morpheus/pull/162)) [@cwharris](https://github.com/cwharris)
- Update mlflow-env.yml ([#146](https://github.com/nv-morpheus/Morpheus/pull/146)) [@pdmack](https://github.com/pdmack)
- Add script to capture triton config ([#116](https://github.com/nv-morpheus/Morpheus/pull/116)) [@pdmack](https://github.com/pdmack)
- Update mlflow-env.yml ([#113](https://github.com/nv-morpheus/Morpheus/pull/113)) [@pdmack](https://github.com/pdmack)
- Jenkins improvememts ([#107](https://github.com/nv-morpheus/Morpheus/pull/107)) [@dagardner-nv](https://github.com/dagardner-nv)
- Rename mlflow conda env file ([#82](https://github.com/nv-morpheus/Morpheus/pull/82)) [@pdmack](https://github.com/pdmack)
- Jenkins integration ([#80](https://github.com/nv-morpheus/Morpheus/pull/80)) [@dagardner-nv](https://github.com/dagardner-nv)
- Revert "Install new apt key" ([#79](https://github.com/nv-morpheus/Morpheus/pull/79)) [@dagardner-nv](https://github.com/dagardner-nv)
- Clear log handlers after each test ([#66](https://github.com/nv-morpheus/Morpheus/pull/66)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix type-o in log parsing example & small formatting fix ([#58](https://github.com/nv-morpheus/Morpheus/pull/58)) [@dagardner-nv](https://github.com/dagardner-nv)

## ⚠️ Known Issues

- Triton 22.04 can crash under heavy load from Morpheus+Kafka ([#259](https://github.com/nv-morpheus/Morpheus/issues/259))


# Morpheus 22.04.00 (27 Apr 2022)

## Initial Public Release

Morpheus is being provided as OSS and is now generally available on GitHub as well as NGC (NVIDIA GPU Cloud). Morpheus is still early software and a work in progress. Breaking changes (including breaking API changes) are to be expected

### Highlights

- GNN (Graph Neural Networking) based workflow for fraud detection
- Transformer based workflow for log parsing
- Updated Morpheus to use the features for pipeline development

## 🐛 Bug Fixes

- Fix default DOCKER_IMAGE_TAG to match that of build_container_release.sh ([#33](https://github.com/nv-morpheus/Morpheus/pull/33)) [@dagardner-nv](https://github.com/dagardner-nv)
- fix incorrect bert vocab and hash in training-tuning-scripts/log-parsing/resources ([#32](https://github.com/nv-morpheus/Morpheus/pull/32)) [@raykallen](https://github.com/raykallen)
- Removing `no_args_is_help` from CLI commands ([#29](https://github.com/nv-morpheus/Morpheus/pull/29)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Log parsing example updates ([#28](https://github.com/nv-morpheus/Morpheus/pull/28)) [@efajardo-nv](https://github.com/efajardo-nv)
- fix to_file "overwrite" option for cli ([#15](https://github.com/nv-morpheus/Morpheus/pull/15)) [@cwharris](https://github.com/cwharris)

## 📖 Documentation

Documentation is provided in the Morpheus GitHub repo (https://github.com/NVIDIA/Morpheus/tree/branch-22.04/docs)

- Update CONTRIBUTING.md & fix file ownership ([#27](https://github.com/nv-morpheus/Morpheus/pull/27)) [@dagardner-nv](https://github.com/dagardner-nv)
- README updates to provide NGC links ([#25](https://github.com/nv-morpheus/Morpheus/pull/25)) [@BartleyR](https://github.com/BartleyR)
- Move model card info to readme and model-information.csv ([#20](https://github.com/nv-morpheus/Morpheus/pull/20)) [@raykallen](https://github.com/raykallen)

## 🛠️ Improvements

- TensorRT installation warning for onnx_to_trt ([#23](https://github.com/nv-morpheus/Morpheus/pull/23)) [@cwharris](https://github.com/cwharris)

## ⚠️ Known Issues

- No known issues
