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

# Morpheus 23.11.01 (7 Dec 2023)

## 🐛 Bug Fixes

- Convert `models/ransomware-models/ransomw-model-short-rf-20220126.sav` to LFS ([#1408](https://github.com/nv-morpheus/Morpheus/pull/1408)) [@mdemoret-nv](https://github.com/mdemoret-nv)

## 📖 Documentation

- Cloud deployment guide fixes ([#1406](https://github.com/nv-morpheus/Morpheus/pull/1406)) [@dagardner-nv](https://github.com/dagardner-nv)


# Morpheus 23.11.00 (30 Nov 2023)

## 🚨 Breaking Changes

- Separate Pipeline type inference/checking &amp; MRC pipeline construction ([#1233](https://github.com/nv-morpheus/Morpheus/pull/1233)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove multiprocess download option ([#1189](https://github.com/nv-morpheus/Morpheus/pull/1189)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🐛 Bug Fixes

- CVE-2023-47248 Mitigation ([#1399](https://github.com/nv-morpheus/Morpheus/pull/1399)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixing the hammah and phishing validation pipelines ([#1398](https://github.com/nv-morpheus/Morpheus/pull/1398)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix the SID Viz workflow shutdown process with the new pipeline shutdown process ([#1392](https://github.com/nv-morpheus/Morpheus/pull/1392)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix race condition in the C++ impl for the pre-process fil stage ([#1390](https://github.com/nv-morpheus/Morpheus/pull/1390)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fixing the conda-build with DOCA enabled and upgrading to CMake 3.25 ([#1386](https://github.com/nv-morpheus/Morpheus/pull/1386)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add missing milvus marker ([#1385](https://github.com/nv-morpheus/Morpheus/pull/1385)) [@dagardner-nv](https://github.com/dagardner-nv)
- Register DataBricksDeltaLakeSourceStage with the CLI ([#1384](https://github.com/nv-morpheus/Morpheus/pull/1384)) [@dagardner-nv](https://github.com/dagardner-nv)
- Guard optional dependencies in try/except blocks ([#1382](https://github.com/nv-morpheus/Morpheus/pull/1382)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix test_vdb_upload_pipe ([#1381](https://github.com/nv-morpheus/Morpheus/pull/1381)) [@dagardner-nv](https://github.com/dagardner-nv)
- DFP container updates ([#1347](https://github.com/nv-morpheus/Morpheus/pull/1347)) [@efajardo-nv](https://github.com/efajardo-nv)
- Removed Mutex Related Milvus Tests ([#1325](https://github.com/nv-morpheus/Morpheus/pull/1325)) [@bsuryadevara](https://github.com/bsuryadevara)
- Pin cuda-python to 11.8.2 as a work around for 11.8.3 incompatibility. ([#1320](https://github.com/nv-morpheus/Morpheus/pull/1320)) [@drobison00](https://github.com/drobison00)
- Forward-merge branch-23.07 to branch-23.11 [resolved conflicts] ([#1246](https://github.com/nv-morpheus/Morpheus/pull/1246)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix tests to support mlflow v2.7 ([#1220](https://github.com/nv-morpheus/Morpheus/pull/1220)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update DFP training model_kwargs ([#1216](https://github.com/nv-morpheus/Morpheus/pull/1216)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix Kafka offset checking test ([#1212](https://github.com/nv-morpheus/Morpheus/pull/1212)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add `review_requested` as a trigger &amp; increased timeouts for camouflage ([#1200](https://github.com/nv-morpheus/Morpheus/pull/1200)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove multiprocess download option ([#1189](https://github.com/nv-morpheus/Morpheus/pull/1189)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update feature length  for test_abp_fil_e2e benchmark ([#1188](https://github.com/nv-morpheus/Morpheus/pull/1188)) [@dagardner-nv](https://github.com/dagardner-nv)
- Make manual_seed an autouse fixture for gnn_fraud_detection_pipeline tests ([#1165](https://github.com/nv-morpheus/Morpheus/pull/1165)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update Downloader utility class to use static dask cluster ([#1161](https://github.com/nv-morpheus/Morpheus/pull/1161)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update to handle GitHub CLI not installed ([#1157](https://github.com/nv-morpheus/Morpheus/pull/1157)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update TimeSeries stage to also work with Production DFP ([#1121](https://github.com/nv-morpheus/Morpheus/pull/1121)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix issue where DFPFileToDataFrameStage logs messages about S3 even when S3 is not in use ([#1120](https://github.com/nv-morpheus/Morpheus/pull/1120)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix Tests broken by Test Reorganization ([#1118](https://github.com/nv-morpheus/Morpheus/pull/1118)) [@cwharris](https://github.com/cwharris)
- Break circular reference issue causing a memory leak ([#1115](https://github.com/nv-morpheus/Morpheus/pull/1115)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix intermittent failures in test_dfencoder_distributed_e2e test ([#1113](https://github.com/nv-morpheus/Morpheus/pull/1113)) [@dagardner-nv](https://github.com/dagardner-nv)
- Resolve forward merger conflices for `branch-23.11` ([#1092](https://github.com/nv-morpheus/Morpheus/pull/1092)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix local CI building from a specific commit ([#1083](https://github.com/nv-morpheus/Morpheus/pull/1083)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Grafana example readme update ([#1393](https://github.com/nv-morpheus/Morpheus/pull/1393)) [@efajardo-nv](https://github.com/efajardo-nv)
- Align model card requirements ([#1388](https://github.com/nv-morpheus/Morpheus/pull/1388)) [@drobison00](https://github.com/drobison00)
- Docs update to indicate use of conda-merge to generate install files ([#1387](https://github.com/nv-morpheus/Morpheus/pull/1387)) [@drobison00](https://github.com/drobison00)
- Stage documentation improvements ([#1362](https://github.com/nv-morpheus/Morpheus/pull/1362)) [@dagardner-nv](https://github.com/dagardner-nv)
- Documentation patch for Examples ([#1357](https://github.com/nv-morpheus/Morpheus/pull/1357)) [@pranavm7](https://github.com/pranavm7)
- Update developer documentation to reflect new compute_schema changes ([#1341](https://github.com/nv-morpheus/Morpheus/pull/1341)) [@dagardner-nv](https://github.com/dagardner-nv)
- Create LICENSE.psycopg2 ([#1295](https://github.com/nv-morpheus/Morpheus/pull/1295)) [@exactlyallan](https://github.com/exactlyallan)
- Fix documentation for morpheus.loaders.sql_loader ([#1264](https://github.com/nv-morpheus/Morpheus/pull/1264)) [@dagardner-nv](https://github.com/dagardner-nv)
- Phishing example fix ([#1215](https://github.com/nv-morpheus/Morpheus/pull/1215)) [@efajardo-nv](https://github.com/efajardo-nv)
- ABP PCAP detection readme update ([#1205](https://github.com/nv-morpheus/Morpheus/pull/1205)) [@efajardo-nv](https://github.com/efajardo-nv)
- Command line examples for module-based DFP pipelines ([#1154](https://github.com/nv-morpheus/Morpheus/pull/1154)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update DFP E2E Benchmarks README to use dev container ([#1125](https://github.com/nv-morpheus/Morpheus/pull/1125)) [@efajardo-nv](https://github.com/efajardo-nv)
- Less intrusive doc builds ([#1060](https://github.com/nv-morpheus/Morpheus/pull/1060)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Add source &amp; stage decorators ([#1364](https://github.com/nv-morpheus/Morpheus/pull/1364)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add a Vector Database Service to allow stages to read and write to VDBs ([#1225](https://github.com/nv-morpheus/Morpheus/pull/1225)) [@bsuryadevara](https://github.com/bsuryadevara)
- CI test stage no longer depends on build stage ([#1219](https://github.com/nv-morpheus/Morpheus/pull/1219)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updates for MRC/Morpheus to build in the same RAPIDS devcontainer environment ([#1171](https://github.com/nv-morpheus/Morpheus/pull/1171)) [@cwharris](https://github.com/cwharris)
- KafkaSourceStage OAuth Callback Support ([#1169](https://github.com/nv-morpheus/Morpheus/pull/1169)) [@cwharris](https://github.com/cwharris)
- GitHub Project Automation and Infra Updates ([#1168](https://github.com/nv-morpheus/Morpheus/pull/1168)) [@jarmak-nv](https://github.com/jarmak-nv)
- Elasticsearch Sink Module ([#1163](https://github.com/nv-morpheus/Morpheus/pull/1163)) [@bsuryadevara](https://github.com/bsuryadevara)
- RSS Source Stage for Reading RSS Feeds ([#1149](https://github.com/nv-morpheus/Morpheus/pull/1149)) [@bsuryadevara](https://github.com/bsuryadevara)
- Add `parser_kwargs` to `FileSourceStage` to support json files ([#1137](https://github.com/nv-morpheus/Morpheus/pull/1137)) [@cwharris](https://github.com/cwharris)
- Add a --viz_direction flag to CLI ([#1119](https://github.com/nv-morpheus/Morpheus/pull/1119)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adds support to read and write to Databricks delta tables ([#630](https://github.com/nv-morpheus/Morpheus/pull/630)) [@pthalasta](https://github.com/pthalasta)

## 🛠️ Improvements

- LLM C++ test and doc updates ([#1379](https://github.com/nv-morpheus/Morpheus/pull/1379)) [@efajardo-nv](https://github.com/efajardo-nv)
- Merge fea-sherlock feature branch into branch-23.11 ([#1359](https://github.com/nv-morpheus/Morpheus/pull/1359)) [@drobison00](https://github.com/drobison00)
- Make dfp_azure_pipeline inference output file configurable. ([#1290](https://github.com/nv-morpheus/Morpheus/pull/1290)) [@drobison00](https://github.com/drobison00)
- Loosen nodejs version restriction ([#1262](https://github.com/nv-morpheus/Morpheus/pull/1262)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use conda environment yaml&#39;s for training-tuning-scripts ([#1256](https://github.com/nv-morpheus/Morpheus/pull/1256)) [@efajardo-nv](https://github.com/efajardo-nv)
- Cherry pick to pull in august DFP enhancements ([#1248](https://github.com/nv-morpheus/Morpheus/pull/1248)) [@drobison00](https://github.com/drobison00)
- [DRAFT] Add model and experiment template &#39;click&#39; options to dfp example pipelines, and make model names Databricks compatible. ([#1245](https://github.com/nv-morpheus/Morpheus/pull/1245)) [@drobison00](https://github.com/drobison00)
- Separate Pipeline type inference/checking &amp; MRC pipeline construction ([#1233](https://github.com/nv-morpheus/Morpheus/pull/1233)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt updated camouflage-server &amp; fix test_dfp_mlflow_model_writer ([#1195](https://github.com/nv-morpheus/Morpheus/pull/1195)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add development deps to dependencies.yaml ([#1193](https://github.com/nv-morpheus/Morpheus/pull/1193)) [@cwharris](https://github.com/cwharris)
- Update to clang-16 &amp; boost-1.82 ([#1186](https://github.com/nv-morpheus/Morpheus/pull/1186)) [@dagardner-nv](https://github.com/dagardner-nv)
- Scope Zookeeper &amp; Kafka fixtures to session ([#1160](https://github.com/nv-morpheus/Morpheus/pull/1160)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use `copy-pr-bot` ([#1159](https://github.com/nv-morpheus/Morpheus/pull/1159)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update dfp_training stage to support ControlMessages or MultiDFPMessages ([#1155](https://github.com/nv-morpheus/Morpheus/pull/1155)) [@drobison00](https://github.com/drobison00)
- Prefer conda package over pip dependencies ([#1135](https://github.com/nv-morpheus/Morpheus/pull/1135)) [@cwharris](https://github.com/cwharris)
- Add tasks and metadata properties to python ControlMessage ([#1134](https://github.com/nv-morpheus/Morpheus/pull/1134)) [@cwharris](https://github.com/cwharris)
- Eliminate redundant code blocks in modules and stages ([#1123](https://github.com/nv-morpheus/Morpheus/pull/1123)) [@bsuryadevara](https://github.com/bsuryadevara)
- update devcontainer base to 23.10 ([#1116](https://github.com/nv-morpheus/Morpheus/pull/1116)) [@cwharris](https://github.com/cwharris)
- Slimmed down CI runners and published artifact urls ([#1112](https://github.com/nv-morpheus/Morpheus/pull/1112)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updating tests to force .pyi files to be committed into the repo ([#1111](https://github.com/nv-morpheus/Morpheus/pull/1111)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- ABP nvsmi sample data generation ([#1108](https://github.com/nv-morpheus/Morpheus/pull/1108)) [@efajardo-nv](https://github.com/efajardo-nv)
- Reorganize C++ Tests ([#1095](https://github.com/nv-morpheus/Morpheus/pull/1095)) [@cwharris](https://github.com/cwharris)
- Improve `gitutils.py` by using the Github CLI when available ([#1088](https://github.com/nv-morpheus/Morpheus/pull/1088)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fixing linting errors which could not be resolved in 23.07 ([#1082](https://github.com/nv-morpheus/Morpheus/pull/1082)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Move testing utilities into tests/_utils ([#1065](https://github.com/nv-morpheus/Morpheus/pull/1065)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update Versions for v23.11.00 ([#1059](https://github.com/nv-morpheus/Morpheus/pull/1059)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update GNN  stellargraph with DGL ([#1032](https://github.com/nv-morpheus/Morpheus/pull/1032)) [@tzemicheal](https://github.com/tzemicheal)
- Implement rest data loader ([#986](https://github.com/nv-morpheus/Morpheus/pull/986)) [@yuchenz427](https://github.com/yuchenz427)
- Adding HTTP sources &amp; sinks ([#977](https://github.com/nv-morpheus/Morpheus/pull/977)) [@dagardner-nv](https://github.com/dagardner-nv)

# Morpheus 23.07.03 (11 Oct 2023)

## 🐛 Bug Fixes
- Add pinned libwebp to resolve CVE ([#1236](https://github.com/nv-morpheus/Morpheus/pull/1236)) [@drobison00](https://github.com/drobison00)
- Add libwebp to meta.yaml for CVE 2307 ([#1242](https://github.com/nv-morpheus/Morpheus/pull/1242)) [@drobison00](https://github.com/drobison00)
- [BUG] Fix Control Message Utils & SQL Max Connections Exhaust ([#1243](https://github.com/nv-morpheus/Morpheus/pull/1243)) [@bsuryadevara](https://github.com/bsuryadevara)

# Morpheus 23.07.02 (25 Jul 2023)

## 🐛 Bug Fixes
- Move data dir to models ([#1099](https://github.com/nv-morpheus/Morpheus/pull/1099)) [@dagardner-nv](https://github.com/dagardner-nv)

# Morpheus 23.07.01 (21 Jul 2023)

## 🐛 Bug Fixes
- Fixing submodules commit ID which got messed up in merge ([#1086](https://github.com/nv-morpheus/Morpheus/pull/1086)) [@mdemoret-nv](https://github.com/mdemoret-nv)

# Morpheus 23.07.00 (20 Jul 2023)

## 🚨 Breaking Changes

- Use size_t for element counts &amp; byte sizes ([#1007](https://github.com/nv-morpheus/Morpheus/pull/1007)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix logging of wrong feature_columns in CLI and exception in auto-complete ([#983](https://github.com/nv-morpheus/Morpheus/pull/983)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use new cudf C++ json writer ([#888](https://github.com/nv-morpheus/Morpheus/pull/888)) [@dagardner-nv](https://github.com/dagardner-nv)
- Python 3.10 support ([#887](https://github.com/nv-morpheus/Morpheus/pull/887)) [@cwharris](https://github.com/cwharris)
- Update to CuDF 23.02 and MRC 23.07 ([#848](https://github.com/nv-morpheus/Morpheus/pull/848)) [@cwharris](https://github.com/cwharris)

## 🐛 Bug Fixes

- Fix fetch_example_data.py to work with s3fs v2303.6 ([#1053](https://github.com/nv-morpheus/Morpheus/pull/1053)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix models scripts/notebooks ([#1051](https://github.com/nv-morpheus/Morpheus/pull/1051)) [@tzemicheal](https://github.com/tzemicheal)
- DFP visualization updates/fixes ([#1043](https://github.com/nv-morpheus/Morpheus/pull/1043)) [@efajardo-nv](https://github.com/efajardo-nv)
- Remove `loop` from several `asyncio` API calls ([#1033](https://github.com/nv-morpheus/Morpheus/pull/1033)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Fix release and dev container builds ([#1027](https://github.com/nv-morpheus/Morpheus/pull/1027)) [@cwharris](https://github.com/cwharris)
- Adopt s3fs 2023.6 per #1022 ([#1023](https://github.com/nv-morpheus/Morpheus/pull/1023)) [@dagardner-nv](https://github.com/dagardner-nv)
- Move hard-coded path in E2E benchmarks to config json ([#1011](https://github.com/nv-morpheus/Morpheus/pull/1011)) [@efajardo-nv](https://github.com/efajardo-nv)
- Use size_t for element counts &amp; byte sizes ([#1007](https://github.com/nv-morpheus/Morpheus/pull/1007)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix pybind11 link errors ([#1003](https://github.com/nv-morpheus/Morpheus/pull/1003)) [@cwharris](https://github.com/cwharris)
- Fixing build with MRC breaking changes ([#998](https://github.com/nv-morpheus/Morpheus/pull/998)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Use cuda-toolkit in release container build ([#997](https://github.com/nv-morpheus/Morpheus/pull/997)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix &quot;No module named &#39;versioneer&#39;&quot; error ([#990](https://github.com/nv-morpheus/Morpheus/pull/990)) [@cwharris](https://github.com/cwharris)
- Warn and cast to cudf, when the C++ impl for MessageMeta receives a pandas DF ([#985](https://github.com/nv-morpheus/Morpheus/pull/985)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix logging of wrong feature_columns in CLI and exception in auto-complete ([#983](https://github.com/nv-morpheus/Morpheus/pull/983)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove duplicate log message in output when using pre-allocation ([#981](https://github.com/nv-morpheus/Morpheus/pull/981)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update to cuda-toolkit in release container build ([#974](https://github.com/nv-morpheus/Morpheus/pull/974)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add override for count method in SlicedMessageMeta ([#972](https://github.com/nv-morpheus/Morpheus/pull/972)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update .bashrc in container to activate morpheus conda environment ([#969](https://github.com/nv-morpheus/Morpheus/pull/969)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix build issues &amp; tests ([#966](https://github.com/nv-morpheus/Morpheus/pull/966)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove/update tests which mock MRC builder objects ([#955](https://github.com/nv-morpheus/Morpheus/pull/955)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove  problematic test causing segfaults ([#954](https://github.com/nv-morpheus/Morpheus/pull/954)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fixing multi-sender stage configurations ([#951](https://github.com/nv-morpheus/Morpheus/pull/951)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add support for fsspec.core.OpenFile instances to the MonitorStage ([#942](https://github.com/nv-morpheus/Morpheus/pull/942)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix race condition in CompareDataFrameStage ([#935](https://github.com/nv-morpheus/Morpheus/pull/935)) [@dagardner-nv](https://github.com/dagardner-nv)
- remove unnecessary quotes from CMAKE_ARGS expression ([#930](https://github.com/nv-morpheus/Morpheus/pull/930)) [@cwharris](https://github.com/cwharris)
- Enforce dtype for ColumnInfo and RenameColumn ([#923](https://github.com/nv-morpheus/Morpheus/pull/923)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix early stopping code in dfencoder to use average loss of batches in validation set ([#908](https://github.com/nv-morpheus/Morpheus/pull/908)) [@hsin-c](https://github.com/hsin-c)
- Fix `get_anomaly_score_losses` in dfencoder to work without categorical features ([#893](https://github.com/nv-morpheus/Morpheus/pull/893)) [@hsin-c](https://github.com/hsin-c)
- Update gnn_fraud_detection_pipeline &amp; ransomware_detection examples to use the same version of dask ([#891](https://github.com/nv-morpheus/Morpheus/pull/891)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Add note about needing docker-compose v1.28+ for DFP ([#1054](https://github.com/nv-morpheus/Morpheus/pull/1054)) [@dagardner-nv](https://github.com/dagardner-nv)
- Clean up CLX references in docstring ([#1049](https://github.com/nv-morpheus/Morpheus/pull/1049)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update docs to use `docker compose` ([#1040](https://github.com/nv-morpheus/Morpheus/pull/1040)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix documentation builds ([#1039](https://github.com/nv-morpheus/Morpheus/pull/1039)) [@dagardner-nv](https://github.com/dagardner-nv)
- GNN FSI modelcard++ ([#1010](https://github.com/nv-morpheus/Morpheus/pull/1010)) [@tzemicheal](https://github.com/tzemicheal)
- DFP Model Card ++ ([#1006](https://github.com/nv-morpheus/Morpheus/pull/1006)) [@hsin-c](https://github.com/hsin-c)
- Document how to build the examples ([#992](https://github.com/nv-morpheus/Morpheus/pull/992)) [@dagardner-nv](https://github.com/dagardner-nv)
- Documentation review edits ([#911](https://github.com/nv-morpheus/Morpheus/pull/911)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add PyTorch to ONNX section to training notebooks ([#903](https://github.com/nv-morpheus/Morpheus/pull/903)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update E2E Benchmarks README ([#880](https://github.com/nv-morpheus/Morpheus/pull/880)) [@efajardo-nv](https://github.com/efajardo-nv)

## 🚀 New Features

- Update to CUDF 23.06 ([#1020](https://github.com/nv-morpheus/Morpheus/pull/1020)) [@cwharris](https://github.com/cwharris)
- Grafana DFP dashboard ([#989](https://github.com/nv-morpheus/Morpheus/pull/989)) [@efajardo-nv](https://github.com/efajardo-nv)
- DFP MultiFileSource optionally poll for file updates ([#978](https://github.com/nv-morpheus/Morpheus/pull/978)) [@dagardner-nv](https://github.com/dagardner-nv)
- Migrate generic components from azure-ad workflow ([#939](https://github.com/nv-morpheus/Morpheus/pull/939)) [@bsuryadevara](https://github.com/bsuryadevara)
- Migrate CLX parsers ([#894](https://github.com/nv-morpheus/Morpheus/pull/894)) [@efajardo-nv](https://github.com/efajardo-nv)
- Python 3.10 support ([#887](https://github.com/nv-morpheus/Morpheus/pull/887)) [@cwharris](https://github.com/cwharris)
- GPUNetIO Integration ([#879](https://github.com/nv-morpheus/Morpheus/pull/879)) [@cwharris](https://github.com/cwharris)
- Update to CuDF 23.02 and MRC 23.07 ([#848](https://github.com/nv-morpheus/Morpheus/pull/848)) [@cwharris](https://github.com/cwharris)

## 🛠️ Improvements

- Support building the Morpheus containers when Morpheus is a submodule ([#1057](https://github.com/nv-morpheus/Morpheus/pull/1057)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Changing GRPC requirement from `grpc-cpp` to `libgrpc` ([#1056](https://github.com/nv-morpheus/Morpheus/pull/1056)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Merge morpheus core spear phishing components. ([#1044](https://github.com/nv-morpheus/Morpheus/pull/1044)) [@drobison00](https://github.com/drobison00)
- Example Documentation Updates ([#1038](https://github.com/nv-morpheus/Morpheus/pull/1038)) [@cwharris](https://github.com/cwharris)
- Update docs and examples to use Triton 23.06 ([#1037](https://github.com/nv-morpheus/Morpheus/pull/1037)) [@efajardo-nv](https://github.com/efajardo-nv)
- SID visualization updates ([#1035](https://github.com/nv-morpheus/Morpheus/pull/1035)) [@efajardo-nv](https://github.com/efajardo-nv)
- New CI images with rapids 23.06 ([#1030](https://github.com/nv-morpheus/Morpheus/pull/1030)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove conda&#39;s base pip (CVE) ([#1024](https://github.com/nv-morpheus/Morpheus/pull/1024)) [@pdmack](https://github.com/pdmack)
- Remove PIP_FIND_LINKS for torch from setup.py ([#1019](https://github.com/nv-morpheus/Morpheus/pull/1019)) [@efajardo-nv](https://github.com/efajardo-nv)
- Upgrade to pytorch 2.0.1 conda package ([#1015](https://github.com/nv-morpheus/Morpheus/pull/1015)) [@efajardo-nv](https://github.com/efajardo-nv)
- Model card ++ for ABP, Phishing, Root cause analysis ([#1014](https://github.com/nv-morpheus/Morpheus/pull/1014)) [@gbatmaz](https://github.com/gbatmaz)
- Bring forward some CVE fixes from 23.03 release ([#1002](https://github.com/nv-morpheus/Morpheus/pull/1002)) [@pdmack](https://github.com/pdmack)
- Remove patch from pybind11 ([#1001](https://github.com/nv-morpheus/Morpheus/pull/1001)) [@dagardner-nv](https://github.com/dagardner-nv)
- Install pytest-kafka via pip/pypi ([#988](https://github.com/nv-morpheus/Morpheus/pull/988)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt MatX 0.4.1 ([#971](https://github.com/nv-morpheus/Morpheus/pull/971)) [@dagardner-nv](https://github.com/dagardner-nv)
- fix_contrib_instructions ([#959](https://github.com/nv-morpheus/Morpheus/pull/959)) [@yuchenz427](https://github.com/yuchenz427)
- Multi-class Sequence Classifier ([#952](https://github.com/nv-morpheus/Morpheus/pull/952)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add Pylint to CI ([#950](https://github.com/nv-morpheus/Morpheus/pull/950)) [@dagardner-nv](https://github.com/dagardner-nv)
- Helper scripts for running CI locally ([#949](https://github.com/nv-morpheus/Morpheus/pull/949)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt updated utilities to pickup MatX 0.4.0 ([#947](https://github.com/nv-morpheus/Morpheus/pull/947)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add COLLABORATOR to exclusion in label-external-issues.yml ([#946](https://github.com/nv-morpheus/Morpheus/pull/946)) [@jarmak-nv](https://github.com/jarmak-nv)
- Multiple Input Kafka Topics Support ([#944](https://github.com/nv-morpheus/Morpheus/pull/944)) [@bsuryadevara](https://github.com/bsuryadevara)
- Tests for stages in DFP Production Example ([#940](https://github.com/nv-morpheus/Morpheus/pull/940)) [@dagardner-nv](https://github.com/dagardner-nv)
- Integrate NVTabular into Morpheus Core and replace existing column_info based workflows. ([#938](https://github.com/nv-morpheus/Morpheus/pull/938)) [@drobison00](https://github.com/drobison00)
- Auto-Comment on External Issues ([#926](https://github.com/nv-morpheus/Morpheus/pull/926)) [@jarmak-nv](https://github.com/jarmak-nv)
- DFP Integrated Training Updates (Stress Testing/Benchmarks) ([#924](https://github.com/nv-morpheus/Morpheus/pull/924)) [@bsuryadevara](https://github.com/bsuryadevara)
- Add CONTRIBUTOR to triage label exception ([#918](https://github.com/nv-morpheus/Morpheus/pull/918)) [@jarmak-nv](https://github.com/jarmak-nv)
- Improve DFP period functionality to allow for better sampling and ignoring period ([#912](https://github.com/nv-morpheus/Morpheus/pull/912)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Tests for gnn_fraud_detection_pipeline &amp; ransomware_detection ([#904](https://github.com/nv-morpheus/Morpheus/pull/904)) [@dagardner-nv](https://github.com/dagardner-nv)
- [ENH] Change external labeler to use the GH CLI for fine-grained token support ([#899](https://github.com/nv-morpheus/Morpheus/pull/899)) [@jarmak-nv](https://github.com/jarmak-nv)
- [ENH] Label External Issues: Update secret, add discussions ([#897](https://github.com/nv-morpheus/Morpheus/pull/897)) [@jarmak-nv](https://github.com/jarmak-nv)
- Use new cudf C++ json writer ([#888](https://github.com/nv-morpheus/Morpheus/pull/888)) [@dagardner-nv](https://github.com/dagardner-nv)
- Create tests for examples with custom stages ([#885](https://github.com/nv-morpheus/Morpheus/pull/885)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use ARC V2 self-hosted runners for GPU jobs ([#878](https://github.com/nv-morpheus/Morpheus/pull/878)) [@jjacobelli](https://github.com/jjacobelli)
- Removing explicit driver install from CI runner ([#877](https://github.com/nv-morpheus/Morpheus/pull/877)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Adding an `update-version.sh` script and CI check to keep versions up to date ([#875](https://github.com/nv-morpheus/Morpheus/pull/875)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add DFP Viz dependencies to DFP container ([#873](https://github.com/nv-morpheus/Morpheus/pull/873)) [@efajardo-nv](https://github.com/efajardo-nv)
- Use eval_batch_size for AutoEncoder loss stats ([#861](https://github.com/nv-morpheus/Morpheus/pull/861)) [@efajardo-nv](https://github.com/efajardo-nv)
- GitHub Infra Update -Delete QST Issue Template in Favor of Discussions, Remove Add to Project Action ([#860](https://github.com/nv-morpheus/Morpheus/pull/860)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add dataset fixture to ease fetching DataFrames for tests ([#847](https://github.com/nv-morpheus/Morpheus/pull/847)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add DataManager object to morpheus/io ([#846](https://github.com/nv-morpheus/Morpheus/pull/846)) [@drobison00](https://github.com/drobison00)
- Suppress volatile warning when building code generated by Cython ([#844](https://github.com/nv-morpheus/Morpheus/pull/844)) [@dagardner-nv](https://github.com/dagardner-nv)
- Replace usage of PreprocessLogParsingStage with PreprocessNLPStage ([#842](https://github.com/nv-morpheus/Morpheus/pull/842)) [@dagardner-nv](https://github.com/dagardner-nv)
- Replace deprecated usage of make_node and make_node_full ([#839](https://github.com/nv-morpheus/Morpheus/pull/839)) [@dagardner-nv](https://github.com/dagardner-nv)
- Bump version to 23.07 and version of MRC dep ([#834](https://github.com/nv-morpheus/Morpheus/pull/834)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update .gitignore to include some common temporary files that shouldn&#39;t be tracked by git ([#829](https://github.com/nv-morpheus/Morpheus/pull/829)) [@dagardner-nv](https://github.com/dagardner-nv)
- Pre-allocate needed columns in abp_pcap_detection example ([#820](https://github.com/nv-morpheus/Morpheus/pull/820)) [@dagardner-nv](https://github.com/dagardner-nv)
- Use ARC V2 self-hosted runners for CPU jobs ([#806](https://github.com/nv-morpheus/Morpheus/pull/806)) [@jjacobelli](https://github.com/jjacobelli)

# Morpheus 23.03.01 (04 Apr 2023)

## 📖 Documentation

- Misc Documentation fixes for 23.03 ([#840](https://github.com/nv-morpheus/Morpheus/pull/840)) [@dagardner-nv](https://github.com/dagardner-nv)

# Morpheus 23.03.00 (30 Mar 2023)

## 🚨 Breaking Changes

- Migrate dfencoder to morpheus repo ([#763](https://github.com/nv-morpheus/Morpheus/pull/763)) [@dagardner-nv](https://github.com/dagardner-nv)
- Improve the python CMake functionality to speed up the configure step ([#754](https://github.com/nv-morpheus/Morpheus/pull/754)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Consolidate size types 684 ([#747](https://github.com/nv-morpheus/Morpheus/pull/747)) [@dagardner-nv](https://github.com/dagardner-nv)
- Deprecate ResponseMemoryProbs &amp; MultiResponseProbsMessage ([#711](https://github.com/nv-morpheus/Morpheus/pull/711)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add Pytest &#39;testpaths&#39; ([#677](https://github.com/nv-morpheus/Morpheus/pull/677)) [@bsuryadevara](https://github.com/bsuryadevara)
- DFP benchmark related pytest error ([#673](https://github.com/nv-morpheus/Morpheus/pull/673)) [@bsuryadevara](https://github.com/bsuryadevara)
- Control MonitorStage output with the log-level ([#659](https://github.com/nv-morpheus/Morpheus/pull/659)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix Python bindings for TensorMemory ([#655](https://github.com/nv-morpheus/Morpheus/pull/655)) [@dagardner-nv](https://github.com/dagardner-nv)
- Table locking &amp; column preallocation ([#586](https://github.com/nv-morpheus/Morpheus/pull/586)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🐛 Bug Fixes

- Add Node.js install instructions to DFP Viz readme ([#828](https://github.com/nv-morpheus/Morpheus/pull/828)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix handling of message offsets in ae pre-processing and timeseries stage ([#821](https://github.com/nv-morpheus/Morpheus/pull/821)) [@dagardner-nv](https://github.com/dagardner-nv)
- Set default DFP container version to 23.03 ([#813](https://github.com/nv-morpheus/Morpheus/pull/813)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix debug log message caused by breaking the long string ([#812](https://github.com/nv-morpheus/Morpheus/pull/812)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix for log_parsing and ransomware_detection examples ([#802](https://github.com/nv-morpheus/Morpheus/pull/802)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix abp_pcap_detection example ([#792](https://github.com/nv-morpheus/Morpheus/pull/792)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix the conda build ([#783](https://github.com/nv-morpheus/Morpheus/pull/783)) [@cwharris](https://github.com/cwharris)
- Fix calls to MultiAEMessage constructor ([#781](https://github.com/nv-morpheus/Morpheus/pull/781)) [@efajardo-nv](https://github.com/efajardo-nv)
- Fix MultiAEMessage constructor ([#780](https://github.com/nv-morpheus/Morpheus/pull/780)) [@efajardo-nv](https://github.com/efajardo-nv)
- Change userid_column_name to match the command line equivelants ([#778](https://github.com/nv-morpheus/Morpheus/pull/778)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix for TritonInferenceStage memory leak ([#776](https://github.com/nv-morpheus/Morpheus/pull/776)) [@cwharris](https://github.com/cwharris)
- Fix intermittent test failure in tests/test_message_meta.py ([#772](https://github.com/nv-morpheus/Morpheus/pull/772)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix absolute depfile causing python package to reinstall each build ([#769](https://github.com/nv-morpheus/Morpheus/pull/769)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Adopt fixed mrc version ([#768](https://github.com/nv-morpheus/Morpheus/pull/768)) [@dagardner-nv](https://github.com/dagardner-nv)
- Have conda-build test cuml installation ([#764](https://github.com/nv-morpheus/Morpheus/pull/764)) [@cwharris](https://github.com/cwharris)
- Document adding cuML and fix the gnn_fraud_detection_pipeline example ([#758](https://github.com/nv-morpheus/Morpheus/pull/758)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add write permission to triage labeler action ([#750](https://github.com/nv-morpheus/Morpheus/pull/750)) [@jarmak-nv](https://github.com/jarmak-nv)
- Replace soon to be deprecated docker base gpuci/miniforge-cuda ([#718](https://github.com/nv-morpheus/Morpheus/pull/718)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix detecting index column when reading from CSV in C++ ([#714](https://github.com/nv-morpheus/Morpheus/pull/714)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add unittest to ensure the kafka source is respecting the pipeline batch size ([#710](https://github.com/nv-morpheus/Morpheus/pull/710)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix DFPTraining validation set option ([#709](https://github.com/nv-morpheus/Morpheus/pull/709)) [@efajardo-nv](https://github.com/efajardo-nv)
- fix incorrect line and byte count benchmarks ([#695](https://github.com/nv-morpheus/Morpheus/pull/695)) [@bsuryadevara](https://github.com/bsuryadevara)
- Warn &amp; replace dataframes with non-unique indexes ([#691](https://github.com/nv-morpheus/Morpheus/pull/691)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update DFPRollingWindowStage to emit correct window ([#683](https://github.com/nv-morpheus/Morpheus/pull/683)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add Pytest &#39;testpaths&#39; ([#677](https://github.com/nv-morpheus/Morpheus/pull/677)) [@bsuryadevara](https://github.com/bsuryadevara)
- DFP benchmark related pytest error ([#673](https://github.com/nv-morpheus/Morpheus/pull/673)) [@bsuryadevara](https://github.com/bsuryadevara)
- Checking out a specific commit, requires a non-shallow checkout ([#662](https://github.com/nv-morpheus/Morpheus/pull/662)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix Python bindings for TensorMemory ([#655](https://github.com/nv-morpheus/Morpheus/pull/655)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Fix source code links from docs ([#774](https://github.com/nv-morpheus/Morpheus/pull/774)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updated and fixed model docs and examples. ([#685](https://github.com/nv-morpheus/Morpheus/pull/685)) [@shawn-davis](https://github.com/shawn-davis)
- Update contributing.md ([#671](https://github.com/nv-morpheus/Morpheus/pull/671)) [@efajardo-nv](https://github.com/efajardo-nv)
- Silence build warnings when building rabbitmq example ([#658](https://github.com/nv-morpheus/Morpheus/pull/658)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Integrated training: Framework updates and Modular DFP pipeline implementation ([#760](https://github.com/nv-morpheus/Morpheus/pull/760)) [@drobison00](https://github.com/drobison00)
- Add support for Parquet file input ([#770](https://github.com/nv-morpheus/Morpheus/pull/770)) [@efajardo-nv](https://github.com/efajardo-nv)
- Add InMemorySinkStage ([#752](https://github.com/nv-morpheus/Morpheus/pull/752)) [@dagardner-nv](https://github.com/dagardner-nv)
- Create a PR template ([#751](https://github.com/nv-morpheus/Morpheus/pull/751)) [@jarmak-nv](https://github.com/jarmak-nv)
- Update to CUDA 11.8 ([#748](https://github.com/nv-morpheus/Morpheus/pull/748)) [@cwharris](https://github.com/cwharris)
- Devcontainer work for #702, #703 ([#717](https://github.com/nv-morpheus/Morpheus/pull/717)) [@cwharris](https://github.com/cwharris)
- Training/Inference Module DFP Production ([#669](https://github.com/nv-morpheus/Morpheus/pull/669)) [@bsuryadevara](https://github.com/bsuryadevara)
- Benchmarks for DFP Production Pipeline ([#664](https://github.com/nv-morpheus/Morpheus/pull/664)) [@bsuryadevara](https://github.com/bsuryadevara)
- Table locking &amp; column preallocation ([#586](https://github.com/nv-morpheus/Morpheus/pull/586)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🛠️ Improvements

- Convert `MonitorStage` to a component ([#805](https://github.com/nv-morpheus/Morpheus/pull/805)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update mlflow base image and python versions for CVE ([#789](https://github.com/nv-morpheus/Morpheus/pull/789)) [@pdmack](https://github.com/pdmack)
- Cleanup all uses of `import morpheus._lib.common` ([#787](https://github.com/nv-morpheus/Morpheus/pull/787)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- End-to-end dfencoder test ([#777](https://github.com/nv-morpheus/Morpheus/pull/777)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add multi-gpu and dataloader support for dfencoder ([#775](https://github.com/nv-morpheus/Morpheus/pull/775)) [@hsin-c](https://github.com/hsin-c)
- Migrate dfencoder to morpheus repo ([#763](https://github.com/nv-morpheus/Morpheus/pull/763)) [@dagardner-nv](https://github.com/dagardner-nv)
- Updating to driver 525 ([#755](https://github.com/nv-morpheus/Morpheus/pull/755)) [@jjacobelli](https://github.com/jjacobelli)
- Improve the python CMake functionality to speed up the configure step ([#754](https://github.com/nv-morpheus/Morpheus/pull/754)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Make use of new MRC_PTR_CAST ([#749](https://github.com/nv-morpheus/Morpheus/pull/749)) [@dagardner-nv](https://github.com/dagardner-nv)
- Consolidate size types 684 ([#747](https://github.com/nv-morpheus/Morpheus/pull/747)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update workflow `permissions` block ([#745](https://github.com/nv-morpheus/Morpheus/pull/745)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use AWS OIDC to get AWS creds ([#742](https://github.com/nv-morpheus/Morpheus/pull/742)) [@jjacobelli](https://github.com/jjacobelli)
- : External issue label ([#739](https://github.com/nv-morpheus/Morpheus/pull/739)) [@jarmak-nv](https://github.com/jarmak-nv)
- Create types.hpp ([#737](https://github.com/nv-morpheus/Morpheus/pull/737)) [@dagardner-nv](https://github.com/dagardner-nv)
- New base test class that invokes an embedded Python interpreter ([#734](https://github.com/nv-morpheus/Morpheus/pull/734)) [@dagardner-nv](https://github.com/dagardner-nv)
- Fix code style and re-enabled IWYU ([#731](https://github.com/nv-morpheus/Morpheus/pull/731)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Avoid copy of source data in  AddClassificationsStage and FilterDetectionsStage ([#730](https://github.com/nv-morpheus/Morpheus/pull/730)) [@dagardner-nv](https://github.com/dagardner-nv)
- Run clang-format on entire repo ([#719](https://github.com/nv-morpheus/Morpheus/pull/719)) [@efajardo-nv](https://github.com/efajardo-nv)
- Deprecate ResponseMemoryProbs &amp; MultiResponseProbsMessage ([#711](https://github.com/nv-morpheus/Morpheus/pull/711)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update `sccache` bucket ([#693](https://github.com/nv-morpheus/Morpheus/pull/693)) [@ajschmidt8](https://github.com/ajschmidt8)
- Move duplicated one-liner to TensorUtils ([#682](https://github.com/nv-morpheus/Morpheus/pull/682)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt v0.12.0 of rabbitmq-c and return to using a shallow checkout ([#679](https://github.com/nv-morpheus/Morpheus/pull/679)) [@dagardner-nv](https://github.com/dagardner-nv)
- Adopt matx v0.3.0 ([#667](https://github.com/nv-morpheus/Morpheus/pull/667)) [@dagardner-nv](https://github.com/dagardner-nv)
- WriteToFileStage optionally emit a flush on each message received ([#663](https://github.com/nv-morpheus/Morpheus/pull/663)) [@dagardner-nv](https://github.com/dagardner-nv)
- Clean up use of find/delete for CVE ([#661](https://github.com/nv-morpheus/Morpheus/pull/661)) [@pdmack](https://github.com/pdmack)
- Control MonitorStage output with the log-level ([#659](https://github.com/nv-morpheus/Morpheus/pull/659)) [@dagardner-nv](https://github.com/dagardner-nv)

# Morpheus 23.01.00 (30 Jan 2023)

## 🚨 Breaking Changes

- Add missing docstrings ([#628](https://github.com/nv-morpheus/Morpheus/pull/628)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove unused `cudf_kwargs` constructor argument from FileSourceStage &amp; NLPVizFileSource ([#602](https://github.com/nv-morpheus/Morpheus/pull/602)) [@dagardner-nv](https://github.com/dagardner-nv)
- DFP: Exclude unwanted columns ([#583](https://github.com/nv-morpheus/Morpheus/pull/583)) [@dagardner-nv](https://github.com/dagardner-nv)
- Morpheus refactor to MRC ([#530](https://github.com/nv-morpheus/Morpheus/pull/530)) [@drobison00](https://github.com/drobison00)
- Add C++ API docs to documentation builds ([#414](https://github.com/nv-morpheus/Morpheus/pull/414)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🐛 Bug Fixes

- Updated gitignore ([#633](https://github.com/nv-morpheus/Morpheus/pull/633)) [@bsuryadevara](https://github.com/bsuryadevara)
- Update meta.yaml ([#621](https://github.com/nv-morpheus/Morpheus/pull/621)) [@pdmack](https://github.com/pdmack)
- Remove hard-coded keys from production DFP ([#607](https://github.com/nv-morpheus/Morpheus/pull/607)) [@efajardo-nv](https://github.com/efajardo-nv)
- Remove unused `cudf_kwargs` constructor argument from FileSourceStage &amp; NLPVizFileSource ([#602](https://github.com/nv-morpheus/Morpheus/pull/602)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update container version to 23.01 in DFP docker-compose ([#578](https://github.com/nv-morpheus/Morpheus/pull/578)) [@dagardner-nv](https://github.com/dagardner-nv)
- rm inadvertent kafka-docker add ([#518](https://github.com/nv-morpheus/Morpheus/pull/518)) [@pdmack](https://github.com/pdmack)
- Fix ambiguous segfault for test requiring MORPHEUS_ROOT ([#514](https://github.com/nv-morpheus/Morpheus/pull/514)) [@cwharris](https://github.com/cwharris)
- Fix offset attr in inference messages ([#513](https://github.com/nv-morpheus/Morpheus/pull/513)) [@dagardner-nv](https://github.com/dagardner-nv)

## 📖 Documentation

- Fix docstring for InferenceMemoryAE ([#653](https://github.com/nv-morpheus/Morpheus/pull/653)) [@dagardner-nv](https://github.com/dagardner-nv)
- 23.01 doc fixes ([#652](https://github.com/nv-morpheus/Morpheus/pull/652)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update NGC version references ([#642](https://github.com/nv-morpheus/Morpheus/pull/642)) [@pdmack](https://github.com/pdmack)
- Updating Code of Conduct ([#640](https://github.com/nv-morpheus/Morpheus/pull/640)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Split up the DFP guide ([#637](https://github.com/nv-morpheus/Morpheus/pull/637)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add md &amp; rst files to copyright.py checks and Fix copyright year in Sphinx footer ([#635](https://github.com/nv-morpheus/Morpheus/pull/635)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add missing docstrings ([#628](https://github.com/nv-morpheus/Morpheus/pull/628)) [@dagardner-nv](https://github.com/dagardner-nv)
- Model and dataset documentation update ([#612](https://github.com/nv-morpheus/Morpheus/pull/612)) [@shawn-davis](https://github.com/shawn-davis)
- Fix production DFP pipeline run commands ([#608](https://github.com/nv-morpheus/Morpheus/pull/608)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update copyright headers for 2023 ([#599](https://github.com/nv-morpheus/Morpheus/pull/599)) [@efajardo-nv](https://github.com/efajardo-nv)
- Ensure Kafka &amp; Triton deps are documented when used ([#598](https://github.com/nv-morpheus/Morpheus/pull/598)) [@dagardner-nv](https://github.com/dagardner-nv)
- Typo(fix): missing equal operator in `--load_model` ([#596](https://github.com/nv-morpheus/Morpheus/pull/596)) [@tanmoyio](https://github.com/tanmoyio)
- Stated explicitly regarding helm chart names ([#592](https://github.com/nv-morpheus/Morpheus/pull/592)) [@bsuryadevara](https://github.com/bsuryadevara)
- Fix spelling mistakes &amp; bad copy/paste ([#590](https://github.com/nv-morpheus/Morpheus/pull/590)) [@dagardner-nv](https://github.com/dagardner-nv)
- Removed outdated comments ([#585](https://github.com/nv-morpheus/Morpheus/pull/585)) [@bsuryadevara](https://github.com/bsuryadevara)
- Document DFP output fields ([#581](https://github.com/nv-morpheus/Morpheus/pull/581)) [@dagardner-nv](https://github.com/dagardner-nv)
- 566 doc source basics overviewrst ([#580](https://github.com/nv-morpheus/Morpheus/pull/580)) [@bsuryadevara](https://github.com/bsuryadevara)
- Small updates to examples/basic_usage/README.md ([#575](https://github.com/nv-morpheus/Morpheus/pull/575)) [@dagardner-nv](https://github.com/dagardner-nv)
- Misc updates for README.md ([#574](https://github.com/nv-morpheus/Morpheus/pull/574)) [@bsuryadevara](https://github.com/bsuryadevara)
- Add a glossary to docs ([#573](https://github.com/nv-morpheus/Morpheus/pull/573)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove search link from index.rst ([#572](https://github.com/nv-morpheus/Morpheus/pull/572)) [@efajardo-nv](https://github.com/efajardo-nv)
- getting_started.md: Goto is two words. ([#571](https://github.com/nv-morpheus/Morpheus/pull/571)) [@dagardner-nv](https://github.com/dagardner-nv)
- Minor fixes/updates to abp_pcap_detection example ([#570](https://github.com/nv-morpheus/Morpheus/pull/570)) [@dagardner-nv](https://github.com/dagardner-nv)
- More documentation fixes ([#560](https://github.com/nv-morpheus/Morpheus/pull/560)) [@dagardner-nv](https://github.com/dagardner-nv)
- Misc documentation fixes ([#547](https://github.com/nv-morpheus/Morpheus/pull/547)) [@dagardner-nv](https://github.com/dagardner-nv)
- Update cloud deployment guide and fix doc build instructions. ([#546](https://github.com/nv-morpheus/Morpheus/pull/546)) [@cwharris](https://github.com/cwharris)
- Move DFP Viz screenshot to LFS ([#537](https://github.com/nv-morpheus/Morpheus/pull/537)) [@efajardo-nv](https://github.com/efajardo-nv)
- Restructure the getting started guide ([#536](https://github.com/nv-morpheus/Morpheus/pull/536)) [@dagardner-nv](https://github.com/dagardner-nv)
- Morpheus not &quot;Morpheus SDK&quot; ([#534](https://github.com/nv-morpheus/Morpheus/pull/534)) [@dagardner-nv](https://github.com/dagardner-nv)
- 525 doc images in docs/source/basics/img should be moved to LFS ([#532](https://github.com/nv-morpheus/Morpheus/pull/532)) [@bsuryadevara](https://github.com/bsuryadevara)
- 524 doc remove usage of buffer stage from examples.rst ([#528](https://github.com/nv-morpheus/Morpheus/pull/528)) [@bsuryadevara](https://github.com/bsuryadevara)
- Rename morpheus_quickstart_guide to cloud_deployment_guide ([#526](https://github.com/nv-morpheus/Morpheus/pull/526)) [@cwharris](https://github.com/cwharris)
- Reorg docs ([#522](https://github.com/nv-morpheus/Morpheus/pull/522)) [@dagardner-nv](https://github.com/dagardner-nv)
- Documentation updates ([#519](https://github.com/nv-morpheus/Morpheus/pull/519)) [@dagardner-nv](https://github.com/dagardner-nv)
- Remove fork references ([#511](https://github.com/nv-morpheus/Morpheus/pull/511)) [@efajardo-nv](https://github.com/efajardo-nv)
- Typo fixes ([#502](https://github.com/nv-morpheus/Morpheus/pull/502)) [@pdmack](https://github.com/pdmack)
- Add C++ API docs to documentation builds ([#414](https://github.com/nv-morpheus/Morpheus/pull/414)) [@dagardner-nv](https://github.com/dagardner-nv)

## 🚀 New Features

- Add GitHub CLI support to devcontainer ([#629](https://github.com/nv-morpheus/Morpheus/pull/629)) [@cwharris](https://github.com/cwharris)
- DFP pipeline module ([#510](https://github.com/nv-morpheus/Morpheus/pull/510)) [@bsuryadevara](https://github.com/bsuryadevara)

## 🛠️ Improvements

- Update protobuf to 3.20.2 ([#648](https://github.com/nv-morpheus/Morpheus/pull/648)) [@pdmack](https://github.com/pdmack)
- Update torch to 1.13.1+cu116 ([#645](https://github.com/nv-morpheus/Morpheus/pull/645)) [@pdmack](https://github.com/pdmack)
- Update morpheus-visualizations submodule ([#639](https://github.com/nv-morpheus/Morpheus/pull/639)) [@efajardo-nv](https://github.com/efajardo-nv)
- Updating the CI image to use Driver 520 ([#634](https://github.com/nv-morpheus/Morpheus/pull/634)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update Rapids version to 22.10 ([#617](https://github.com/nv-morpheus/Morpheus/pull/617)) [@efajardo-nv](https://github.com/efajardo-nv)
- Update DFPMLFlowModelWriterStage to no longer save mean and std metrics ([#605](https://github.com/nv-morpheus/Morpheus/pull/605)) [@efajardo-nv](https://github.com/efajardo-nv)
- Log parsing model training for split logs and smaller models ([#597](https://github.com/nv-morpheus/Morpheus/pull/597)) [@raykallen](https://github.com/raykallen)
- Ensure that the mlflow logger is uses at the same level as the morpheus logger ([#594](https://github.com/nv-morpheus/Morpheus/pull/594)) [@dagardner-nv](https://github.com/dagardner-nv)
- DFP: Exclude unwanted columns ([#583](https://github.com/nv-morpheus/Morpheus/pull/583)) [@dagardner-nv](https://github.com/dagardner-nv)
- Add validation set option to DFP training ([#579](https://github.com/nv-morpheus/Morpheus/pull/579)) [@efajardo-nv](https://github.com/efajardo-nv)
- Updating CI runner image for MRC and C++20 ([#556](https://github.com/nv-morpheus/Morpheus/pull/556)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add Morpheus devcontainer ([#535](https://github.com/nv-morpheus/Morpheus/pull/535)) [@cwharris](https://github.com/cwharris)
- Morpheus changes related to utility consolidation ([#531](https://github.com/nv-morpheus/Morpheus/pull/531)) [@drobison00](https://github.com/drobison00)
- Morpheus refactor to MRC ([#530](https://github.com/nv-morpheus/Morpheus/pull/530)) [@drobison00](https://github.com/drobison00)
- Update to PyTorch 1.12 ([#523](https://github.com/nv-morpheus/Morpheus/pull/523)) [@efajardo-nv](https://github.com/efajardo-nv)

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
