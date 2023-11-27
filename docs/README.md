<!--
 SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Building Documentation

Additional packages required for building the documentation are defined in `./conda_docs.yml`.

## Install Additional Dependencies
From the root of the Morpheus repo:
```bash
export CUDA_VER=11.8
mamba install -n base -c conda-forge conda-merge
conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml \
  docs/conda_docs.yml > .tmp/merged.yml \
  && mamba env update -n ${CONDA_DEFAULT_ENV} --file .tmp/merged.yml
```

## Build Morpheus and Documentation
```
CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_DOCS=ON" ./scripts/compile.sh --target morpheus_docs
```
Outputs to `build/docs/html`
