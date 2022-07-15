# Morpheus 22.06.001 (15 Jul 2022)

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
